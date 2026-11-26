// Teensy 4.1 + Rev D Audio Shield (left LINE IN) + Adafruit ISM330DHCX.
// USB Type: Serial. Install Adafruit LSM6DS and its dependencies.
// S starts preview; X stops. The desktop app records a slice of this stream.
#include <Audio.h>
#include <Wire.h>
#include <Adafruit_ISM330DHCX.h>

constexpr uint8_t IMU_ADDRESS = 0x6A; // Change to 0x6B only if SA0 is high.
// Accelerometer output data rate: 833, 1666 or 3332 (nominal Hz; the sensor's clock runs
// ~8% fast, e.g. 833 -> ~904 Hz measured). 1666 doubles the vibration bandwidth (useful
// for cough vs throat clear); the host averages back to ~900 Hz for existing models.
constexpr unsigned IMU_ODR_HZ = 1666;
constexpr float IMU_RATE = IMU_ODR_HZ;
constexpr lsm6ds_data_rate_t IMU_ODR_SETTING =
    IMU_ODR_HZ >= 3332 ? LSM6DS_RATE_3_33K_HZ : IMU_ODR_HZ >= 1666 ? LSM6DS_RATE_1_66K_HZ : LSM6DS_RATE_833_HZ;
// Allow two sample periods between reads before reporting a possibly overwritten sample.
constexpr uint32_t IMU_MAX_GAP_US = (uint32_t)(2.0f * 1e6f / IMU_RATE);
// ISM330DHCX at +/-2 g: 0.061 mg per LSB (datasheet), converted to m/s^2.
constexpr float ACCEL_SCALE = 0.061e-3f * 9.80665f;
constexpr uint8_t MIC_CHANNEL = 0;   // Left line input; use 1 for right.
constexpr unsigned QUEUE_SIZE = 32;

struct MicBlock {
  uint32_t index;
  uint32_t timeUs;
  int16_t samples[AUDIO_BLOCK_SAMPLES];
};

// A small single-producer/single-consumer ring. The audio interrupt never
// prints or waits for USB. A full queue is an explicit error, never hidden.
class TimestampedAudio : public AudioStream {
 public:
  TimestampedAudio() : AudioStream(1, inputQueue) {}
  volatile bool enabled = false;
  volatile bool overflow = false;

  void begin() {
    AudioNoInterrupts();
    head = tail = 0;
    sampleIndex = 0;
    skipFirst = true; // Discard the block straddling the start command.
    overflow = false;
    enabled = true;
    AudioInterrupts();
  }

  void end() { enabled = false; }

  bool pop(MicBlock &destination) {
    if (tail == head) return false;
    destination = ring[tail];
    __DMB();
    tail = (tail + 1) % QUEUE_SIZE;
    return true;
  }

  void update() override {
    audio_block_t *block = receiveReadOnly();
    if (!block) {
      if (enabled) overflow = true;
      return;
    }
    if (enabled) {
      if (skipFirst) {
        skipFirst = false;
        firstTime = micros(); // Next block starts at this approximate time.
      } else {
        unsigned next = (head + 1) % QUEUE_SIZE;
        if (next == tail) {
          overflow = true;
        } else {
          MicBlock &item = ring[head];
          item.index = sampleIndex;
          item.timeUs = firstTime + (uint32_t)((double)sampleIndex * 1e6 / AUDIO_SAMPLE_RATE_EXACT);
          memcpy(item.samples, block->data, sizeof(item.samples));
          __DMB();
          head = next;
        }
        sampleIndex += AUDIO_BLOCK_SAMPLES;
      }
    }
    release(block);
  }

 private:
  audio_block_t *inputQueue[1];
  MicBlock ring[QUEUE_SIZE];
  volatile unsigned head = 0, tail = 0;
  uint32_t sampleIndex = 0, firstTime = 0;
  bool skipFirst = true;
};

AudioInputI2S audioInput;
TimestampedAudio audioQueue;
AudioConnection connection(audioInput, MIC_CHANNEL, audioQueue, 0);
AudioControlSGTL5000 audioShield;
Adafruit_ISM330DHCX imu;
bool hardwareReady = false, streaming = false, pendingAudio = false;
uint32_t imuIndex = 0, lastImuRead = 0, streamStart = 0;
MicBlock outgoing;
char audioLine[1100];
size_t audioLength = 0;

void fail(const char *message) {
  audioQueue.end();
  streaming = false;
  Serial.print("E,");
  Serial.println(message);
}

void setup() {
  Serial.begin(115200); // Native USB serial; baud does not throttle USB.
  AudioMemory(16);
  Wire.begin(); // SDA 18, SCL 19 shared with the audio shield.
  if (!audioShield.enable()) return;
  audioShield.inputSelect(AUDIO_INPUT_LINEIN);
  audioShield.lineInLevel(5); // Fixed gain. Check clipping before participant use.
  audioShield.muteHeadphone();
  if (!imu.begin_I2C(IMU_ADDRESS, &Wire)) return;
  // 1 MHz (Fast-mode Plus, supported by the ISM330DHCX) leaves headroom for 1.66-3.33 kHz.
  // The SGTL5000 is only addressed during setup above, at the default speed.
  Wire.setClock(IMU_ODR_HZ > 833 ? 1000000 : 400000);
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_2_G);
  imu.setAccelDataRate(IMU_ODR_SETTING);
  imu.setGyroDataRate(LSM6DS_RATE_SHUTDOWN);
  hardwareReady = true;
}

// Read OUTX_L_A..OUTZ_H_A (0x28-0x2D) directly: 6 bytes instead of the 14 that
// getEvent() reads (gyro and temperature are unused), so faster rates keep up.
bool readAccel(float &x, float &y, float &z) {
  Wire.beginTransmission(IMU_ADDRESS);
  Wire.write(0x28);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(IMU_ADDRESS, (uint8_t)6) != 6) return false;
  int16_t raw[3];
  for (int16_t &value : raw) {
    uint8_t low = Wire.read(), high = Wire.read();
    value = (int16_t)((high << 8) | low);
  }
  x = raw[0] * ACCEL_SCALE; y = raw[1] * ACCEL_SCALE; z = raw[2] * ACCEL_SCALE;
  return true;
}

bool accelReady() {
  // STATUS_REG.XLDA: read only when a new accelerometer sample is available.
  Wire.beginTransmission(IMU_ADDRESS);
  Wire.write(0x1E);
  if (Wire.endTransmission(false) != 0) {
    fail("IMU I2C connection lost");
    return false;
  }
  if (Wire.requestFrom(IMU_ADDRESS, (uint8_t)1) != 1) {
    fail("IMU status read failed");
    return false;
  }
  return Wire.read() & 1;
}

void loop() {
  while (Serial.available()) {
    char command = Serial.read();
    if (command == 'X') {
      audioQueue.end();
      streaming = false;
      pendingAudio = false;
    } else if (command == 'S') {
      if (!hardwareReady) {
        fail("Audio shield or IMU not found; check power and wiring");
        continue;
      }
      audioQueue.end();
      pendingAudio = false;
      imuIndex = 0;
      streamStart = lastImuRead = micros();
      Serial.print("S,"); Serial.print(streamStart);
      Serial.print(','); Serial.print(AUDIO_SAMPLE_RATE_EXACT, 6);
      Serial.print(','); Serial.print(IMU_RATE, 3);
      Serial.println(",m/s2,V1");
      audioQueue.begin();
      streaming = true;
    }
  }
  if (!streaming) return;
  if (!Serial) { audioQueue.end(); streaming = false; return; }
  if ((uint32_t)(micros() - streamStart) > 3600000000UL) {
    fail("Preview reached one hour; reconnect"); return;
  }
  if (audioQueue.overflow) { fail("Audio buffer overrun or missing audio block"); return; }

  if (accelReady()) {
    uint32_t timestamp = micros(); // Acquisition-read time, not USB-send time.
    if (imuIndex > 0 && (uint32_t)(timestamp - lastImuRead) > IMU_MAX_GAP_US) {
      fail("IMU service delay; possible overwritten sample"); return;
    }
    float ax, ay, az;
    if (!readAccel(ax, ay, az)) {
      fail("IMU sample read failed"); return;
    }
    lastImuRead = timestamp;
    char line[160];
    int length = snprintf(line, sizeof(line), "I,%lu,%lu,%.7f,%.7f,%.7f\n",
                          (unsigned long)imuIndex++, (unsigned long)timestamp, ax, ay, az);
    if (Serial.availableForWrite() < length) { fail("USB cannot keep up with IMU"); return; }
    Serial.write((const uint8_t *)line, length);
  }
  if (!streaming) return;
  if ((uint32_t)(micros() - lastImuRead) > 100000) {
    fail("No new IMU samples"); return;
  }
  if (!pendingAudio && audioQueue.pop(outgoing)) {
    int used = snprintf(audioLine, sizeof(audioLine), "A,%lu,%lu",
                        (unsigned long)outgoing.index, (unsigned long)outgoing.timeUs);
    for (unsigned i = 0; i < AUDIO_BLOCK_SAMPLES; ++i) {
      used += snprintf(audioLine + used, sizeof(audioLine) - used, ",%d", outgoing.samples[i]);
    }
    audioLine[used++] = '\n';
    audioLength = used;
    pendingAudio = true;
  }
  if (pendingAudio && Serial.availableForWrite() >= (int)audioLength) {
    Serial.write((const uint8_t *)audioLine, audioLength);
    pendingAudio = false;
  }
}
