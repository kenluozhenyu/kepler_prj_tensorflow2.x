"""
Kepler CNN Model Implementation (TensorFlow 2.x)

This module implements a dual-input CNN architecture for Kepler exoplanet detection.
The model processes both global and local views of light curve data to classify
exoplanet candidates.

Architecture:
- Global view branch: 2001-point time series with deeper convolutional layers
- Local view branch: 201-point time series with focused convolutional processing
- Merge branch: Combined features processed through dense layers with dropout
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from configure import environment


class KeplerCNN:
    """
    Kepler CNN model class with TensorFlow 2.x best practices.

    Features:
    - Dual-input architecture for global and local views
    - Modern Keras subclassing API
    - Proper regularization and initialization
    - Flexible model configuration
    """

    def __init__(self,
                 global_view_shape=(2001, 1),
                 local_view_shape=(201, 1),
                 dropout_rate=0.5,
                 l2_reg=1e-4):
        """
        Initialize Kepler CNN model.

        Args:
            global_view_shape: Shape of global view input (2001, 1)
            local_view_shape: Shape of local view input (201, 1)
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
        """
        self.global_view_shape = global_view_shape
        self.local_view_shape = local_view_shape
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.num_classes = environment.NB_CLASSES

    def _build_global_branch(self):
        """
        Build the global view processing branch.

        Returns:
            tf.keras.Model: Global view branch model
        """
        inputs = layers.Input(shape=self.global_view_shape, name='global_input')

        # Block 1: 16 filters
        x = layers.Conv1D(16, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv1_1')(inputs)
        x = layers.Conv1D(16, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv1_2')(x)
        x = layers.MaxPooling1D(5, 2, name='global_pool1')(x)
        x = layers.BatchNormalization(name='global_bn1')(x)

        # Block 2: 32 filters
        x = layers.Conv1D(32, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv2_1')(x)
        x = layers.Conv1D(32, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv2_2')(x)
        x = layers.MaxPooling1D(5, 2, name='global_pool2')(x)
        x = layers.BatchNormalization(name='global_bn2')(x)

        # Block 3: 64 filters
        x = layers.Conv1D(64, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv3_1')(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv3_2')(x)
        x = layers.MaxPooling1D(5, 2, name='global_pool3')(x)
        x = layers.BatchNormalization(name='global_bn3')(x)

        # Block 4: 128 filters
        x = layers.Conv1D(128, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv4_1')(x)
        x = layers.Conv1D(128, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv4_2')(x)
        x = layers.MaxPooling1D(5, 2, name='global_pool4')(x)
        x = layers.BatchNormalization(name='global_bn4')(x)

        # Block 5: 256 filters
        x = layers.Conv1D(256, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv5_1')(x)
        x = layers.Conv1D(256, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='global_conv5_2')(x)
        x = layers.MaxPooling1D(5, 2, name='global_pool5')(x)
        x = layers.BatchNormalization(name='global_bn5')(x)

        # Flatten for merging
        x = layers.Flatten(name='global_flatten')(x)

        return models.Model(inputs=inputs, outputs=x, name='global_branch')

    def _build_local_branch(self):
        """
        Build the local view processing branch.

        Returns:
            tf.keras.Model: Local view branch model
        """
        inputs = layers.Input(shape=self.local_view_shape, name='local_input')

        # Block 1: 16 filters
        x = layers.Conv1D(16, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='local_conv1_1')(inputs)
        x = layers.Conv1D(16, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='local_conv1_2')(x)
        x = layers.MaxPooling1D(7, 2, name='local_pool1')(x)
        x = layers.BatchNormalization(name='local_bn1')(x)

        # Block 2: 32 filters
        x = layers.Conv1D(32, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='local_conv2_1')(x)
        x = layers.Conv1D(32, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(self.l2_reg),
                         name='local_conv2_2')(x)
        x = layers.MaxPooling1D(7, 2, name='local_pool2')(x)
        x = layers.BatchNormalization(name='local_bn2')(x)

        # Flatten for merging
        x = layers.Flatten(name='local_flatten')(x)

        return models.Model(inputs=inputs, outputs=x, name='local_branch')

    def _build_classifier(self, merged_features):
        """
        Build the classifier part of the model.

        Args:
            merged_features: Merged features from global and local branches

        Returns:
            tf.Tensor: Output tensor
        """
        # Dense layers with dropout and batch normalization
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name='dense1')(merged_features)
        x = layers.BatchNormalization(name='dense1_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout1')(x)

        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name='dense2')(x)
        x = layers.BatchNormalization(name='dense2_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout2')(x)

        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name='dense3')(x)
        x = layers.BatchNormalization(name='dense3_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout3')(x)

        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name='dense4')(x)
        x = layers.BatchNormalization(name='dense4_bn')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout4')(x)

        # Output layer
        if self.num_classes == 2:
            activation = 'sigmoid'
            units = 1
        else:
            activation = 'softmax'
            units = self.num_classes

        output = layers.Dense(units, activation=activation,
                            name='predictions')(x)

        return output

    def build_model(self):
        """
        Build the complete Kepler CNN model.

        Returns:
            tf.keras.Model: Complete model
        """
        # Build branches
        global_branch = self._build_global_branch()
        local_branch = self._build_local_branch()

        # Get inputs
        global_input = global_branch.input
        local_input = local_branch.input

        # Get outputs
        global_features = global_branch.output
        local_features = local_branch.output

        # Merge features
        merged = layers.concatenate([global_features, local_features],
                                  name='merge_features')

        # Build classifier
        output = self._build_classifier(merged)

        # Create model
        model = models.Model(
            inputs=[global_input, local_input],
            outputs=output,
            name='KeplerCNN'
        )

        return model

    def build_compiled_model(self, learning_rate=1e-5, decay=1e-6):
        """
        Build and compile the model with appropriate loss and metrics.

        Args:
            learning_rate: Learning rate for optimizer
            decay: Learning rate decay

        Returns:
            tf.keras.Model: Compiled model
        """
        model = self.build_model()

        # Choose optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            decay=decay
        )

        # Choose loss and metrics based on number of classes
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
            if self.num_classes >= 10:
                metrics.append('top_k_categorical_accuracy')

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return model


def build_Kepler_CNN():
    """
    Legacy function for backward compatibility.

    Returns:
        tf.keras.Model: Kepler CNN model (not compiled)
    """
    cnn = KeplerCNN()
    return cnn.build_model()


def build_compiled_Kepler_CNN(learning_rate=1e-5, decay=1e-6):
    """
    Build and compile Kepler CNN model.

    Args:
        learning_rate: Learning rate for optimizer
        decay: Learning rate decay

    Returns:
        tf.keras.Model: Compiled model
    """
    cnn = KeplerCNN()
    return cnn.build_compiled_model(learning_rate, decay)


if __name__ == "__main__":
    # Test model creation
    print("Building Kepler CNN model...")

    # Create model instance
    cnn = KeplerCNN()

    # Build model
    model = cnn.build_model()

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Test with dummy data
    print("\nTesting model with dummy data...")
    dummy_global = tf.random.normal((1, 2001, 1))
    dummy_local = tf.random.normal((1, 201, 1))

    predictions = model([dummy_global, dummy_local])
    print(f"Output shape: {predictions.shape}")
    print(f"Output range: [{tf.reduce_min(predictions):.3f}, {tf.reduce_max(predictions):.3f}]")

    # Test compiled model
    print("\nTesting compiled model...")
    compiled_model = cnn.build_compiled_model()

    # Test prediction with compiled model
    compiled_predictions = compiled_model([dummy_global, dummy_local])
    print(f"Compiled model output shape: {compiled_predictions.shape}")

    print("\nModel creation successful!")