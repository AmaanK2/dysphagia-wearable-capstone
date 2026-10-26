# Hardware integration

The proposed prototype records neck sound and motion and sends data to a laptop. Hardware selection and acquisition integration are not implemented here. EMG was discussed but is not an input to the current models.

Before writing an adapter, record the selected sensor, mounting position, units, channel order, measured sampling rate, timestamp format and synchronization method. Keep a short permitted example recording and its acquisition settings locally for integration testing. Do not assume the public-dataset models transfer without validation on the actual wearable.
