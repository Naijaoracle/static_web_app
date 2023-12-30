from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging


def create_model():
    model = Sequential()
    try:
        # Define your model architecture here
        # Make sure to define the same architecture as the one used during training
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(224, 224, 3), kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.6))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logging.error(f"An error occurred while creating the model: {str(e)}")
        return None

def load_model_weights_from_blob(blob_url, connection_string):
    try:
        from azure.storage.blob import BlobClient
        import io
        import os
        
        connection_string = os.environ["AzureWebJobsStorage"]
        blob_client = BlobClient.from_blob_url(blob_url, connection_string=connection_string)
        download_stream = blob_client.download_blob().readall()
        
        model = create_model()
        if model is not None:
            model.load_weights(io.BytesIO(download_stream))
        return model
    except Exception as e:
        logging.error(f"An error occurred while loading model weights: {str(e)}")
        return None
