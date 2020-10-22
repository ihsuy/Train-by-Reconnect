import numpy as np
import random
import tensorflow as tf

# Use option 1 if you do not run this script in a Jupyter notebook
# from tqdm import trange as pbar # Option 1
from tqdm.notebook import trange as pbar  # Option 2


class LaPerm:
    """Lookahead Permutation　(LaPerm). 
    """
    def __init__(self, skip_bn, skip_bias):
        """ Initialize LaPerm algorithm with hyperparameters.
        Args:
            Please refer to the LaPermTrainLoop class.
        """
        self.skip_bn = skip_bn
        self.skip_bias = skip_bias
        self._model_set = False

    def set_model(self, model):
        """Get a reference to the model."""
        self._model = model
        self._model_set = True

    def sort_theta(self):
        """Create theta_sorted.
        Creates a copy of theta_0 (the initial, e.g., random weights) 
        and ascendingly sorts every weight vector of this copy.
        The outcome is then stored as theta_sorted.
        """
        if not self._model_set:
            raise AssertionError(
                "Can't create slots without first calling self.set_model.")

        self.theta_sorted = []
        for var in self._model.trainable_variables:
            var_rank = tf.rank(var)
            if var_rank == 1:
                var = self.sort_1d(var)
            elif var_rank == 2:
                var = self.sort_2d(var)
            elif var_rank == 4:
                var = self.sort_4d(var)
            else:
                raise NotImplementedError(
                    "Unknown tensor rank: {}".format(var_rank))
            self.theta_sorted.append(var)

    @tf.function
    def apply_synchronization(self, theta_t):
        """Synchronization.
        Permute every weight matrix theta_sorted_i in theta_sorted according to
        its counterpart theta_t_i in theta_t.

        Args: 
            theta_t - (list of tensors) - weight matrices of weight vectors
                w_j to be synchronized
        """

        if not self._model_set:
            raise AssertionError(
                "Can't apply gradients without first calling self.set_model.")

        for theta_t_i, theta_sorted_i in zip(theta_t, self.theta_sorted):
            is_bn = "batch_normalization" in theta_t_i.name
            is_bias = "bias" in theta_t_i.name
            if (self.skip_bn and is_bn) or (self.skip_bias and is_bias):
                continue

            # rank = tf.rank(theta_t_i)
            theta_permuted = theta_t_i
            if theta_t_i.shape.ndims == 1:
                theta_permuted = self.synchronize_1d(theta_t_i, theta_sorted_i)
            elif theta_t_i.shape.ndims == 2:
                theta_permuted = self.synchronize_2d(theta_t_i, theta_sorted_i)
            elif theta_t_i.shape.ndims == 4:
                theta_permuted = self.synchronize_4d(theta_t_i, theta_sorted_i)
            else:
                raise NotImplementedError(
                    "Unknown tensor rank: {}".format(theta_t_i.shape.ndims))
            theta_t_i.assign(theta_permuted)

    @tf.function
    def synchronize_1d(self, var, var_sorted):
        """ Synchronize var (if ndim(var) == 1)
        Please refer to synchronize_4d.
        """
        old_shape = var.shape
        var_shape = [1, var.shape[0]]
        var = self.make_vectors_1d(var)
        permuted = self.permute(var, var_shape, var_sorted)
        return self.make_variable_1d(permuted, old_shape)

    @tf.function
    def synchronize_2d(self, var, var_sorted):
        """ Synchronize var (if ndim(var) == 2)
        Please refer to synchronize_4d.
        """
        old_shape = var.shape
        var_shape = [var.shape[1], var.shape[0]]
        var = self.make_vectors_2d(var)
        permuted = self.permute(var, var_shape, var_sorted)
        return self.make_variable_2d(permuted, old_shape)

    @tf.function
    def synchronize_4d(self, var, var_sorted):
        """ Synchronize var (if ndim(var) == 4)
        (1) transforms var into a 2d weight matrix where each row is 
        an 1d weight vector. (2) produce a permuted version of var_sorted 
        against this weight matrix obtained in (1). 
        (3) transforms the permuted var_sorted back to the shape of var.
        """
        old_shape = var.shape
        var_shape = [var.shape[3], var.shape[0]*var.shape[1]*var.shape[2]]
        var = self.make_vectors_4d(var)
        permuted = self.permute(var, var_shape, var_sorted)
        return self.make_variable_4d(permuted, old_shape)

    def permute(self, var, var_shape, var_sorted):
        """ Permute var_sorted w.r.t. the rankings of var, such that
        weight vectors w_j in var and w_j_prime in var_sorted has the 
        same ranking for all j.

        Args:
            var - (2d tensor) - a tensor where each row is a 
                weight vector to be permuted
            var_shape - (tensor) - the shape of var
            var_sorted - (2d tensor) - the weight matrix to 
                be permuted against var.
        """
        ranking = tf.argsort(tf.argsort(var))
        row_idx = tf.transpose(tf.repeat(tf.expand_dims(
            tf.range(0, var_shape[0]), 0), [var_shape[1]], axis=0))
        pair = tf.stack([row_idx, ranking])
        permutation_idx = tf.transpose(
            pair, perm=tf.convert_to_tensor([1, 2, 0]))
        # Perform permutation by indexing
        return tf.gather_nd(var_sorted, permutation_idx)

    def make_vectors_1d(self, var):
        """ Please refer to make_vectors_4d. 
        """
        return tf.expand_dims(var, axis=0)

    def make_vectors_2d(self, var):
        """ Please refer to make_vectors_4d.
        """
        return tf.transpose(var)

    def make_vectors_4d(self, var):
        """ Transforms var into 2d matrix where each 
        row is a weight vector. 
        """
        w = tf.transpose(var, perm=tf.convert_to_tensor([3, 0, 1, 2]))
        return tf.reshape(w, shape=tf.convert_to_tensor([w.shape[0], -1]))

    def make_variable_1d(self, var, oldshape):
        """ Please refer to make_variable_4d. 
        """
        return var[0]

    def make_variable_2d(self, var, oldshape):
        """ Please refer to make_variable_4d. 
        """
        return tf.transpose(var)

    def make_variable_4d(self, var, oldshape):
        """ Revert the effect of make_vectors_4d(var).
        """
        shape = tf.convert_to_tensor(
            [oldshape[3], oldshape[0], oldshape[1], oldshape[2]])
        return tf.transpose(tf.reshape(var, shape=shape), perm=tf.convert_to_tensor([1, 2, 3, 0]))

    @tf.function
    def sort_1d(self, var):
        return tf.sort(self.make_vectors_1d(var))

    @tf.function
    def sort_2d(self, var):
        return tf.sort(self.make_vectors_2d(var))

    @tf.function
    def sort_4d(self, var):
        return tf.sort(self.make_vectors_4d(var))


class LaPermTrainLoop:
    """ Train loop which runs LaPerm on a deep neural network model.

    Args:
            model - (tensorflow.python.keras.Model) - A model instance
                which supports access/modification of its trainable variables.
            loss: - (Str or tensorflow.losses instance) - The loss function 
                used to compare the model output and result.
            inner_optimizer - (tensorflow.keras.optimizers instance) - The 
                inner optimizer of LaPerm. Please refer to Section 4. 
            k_schedule - (callable) - returns synchronization period k
                 given the number of epochs.
                 Example: 
                    def k_schedule(epochs):
                        k = 20
                        if epochs > 50:
                            k*=5
                        return k
            lr_schedule - (callable) - returns learning rate given the 
                number of epochs.
                Example:
                    def lr_schedule(epoch):
                        lr = 1e-3
                        if epoch > 50:
                            lr = 1e-4
                        if epoch > 85:
                            lr = 1e-5
                        return lr
            skip_bn - (boolean) - Whether to apply synchronization on 
                batch normalization layers. If True, the batch normalization
                layers are trained only using the inner optimizer (no 
                permutation will be performed). 
            skip_bias - (boolean) - Whether to apply synchronization on 
                biases. If True, the biases are trained only using the 
                inner optimizer (no permutation will be performed). 
    """

    def __init__(self,
                 model,
                 loss,
                 inner_optimizer,
                 k_schedule,
                 lr_schedule,
                 skip_bn=True,
                 skip_bias=True):
        self._model = model
        self.iterations = 0

        if isinstance(loss, str):
            try:
                self._loss = getattr(tf.keras.losses, loss)
            except:
                raise Exception("Undefined loss function: {}".format(loss))

        self.LaPerm = LaPerm(skip_bn=skip_bn, skip_bias=skip_bias)
        self.LaPerm.set_model(self._model)

        self.inner_optimizer = inner_optimizer
        self.k_schedule = k_schedule
        self.lr_schedule = lr_schedule

    @tf.function
    def loss(self, y_true, y_pred):
        return self._loss(y_true=y_true, y_pred=y_pred)

    @tf.function
    def grad(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(self._model.trainable_variables)
            y_pred = self._model(x, training=True)
            loss_value = self.loss(y_true=y, y_pred=y_pred)
            loss_value = tf.add(loss_value, sum(self._model.losses))

        grads = tape.gradient(loss_value, self._model.trainable_variables)

        return loss_value, grads

    def evaluate(self, x, y, batch=1000):
        """Returns the loss and accuracy for the model in test mode.
        Args:
            x - (list of arrays) - Input data.
            y - (array) - Target data.
            batch - (int or None) - Number of samples per batch to 
                calculate the result.
        """
        if not batch:
            batch = len(x)

        accuracy_fn = tf.metrics.Accuracy()
        losses = []

        total = len(x)
        loc, last_loc = 0, total-batch
        while loc < total:
            if loc > last_loc:
                batch = total-loc

            x_ = x[loc:loc+batch]
            y_ = y[loc:loc+batch]
            y_pred = self._model(x_, training=False)

            loss_value = self.loss(y_true=y_, y_pred=y_pred)
            loss_value = tf.add(loss_value, sum(self._model.losses))
            losses += list(loss_value)

            # If y_ is onehot
            if len(y_.shape) > 1 and y_.shape[1] != 1:
                y_ = tf.argmax(y_, axis=1)

            pred = tf.argmax(y_pred, axis=1)
            accuracy_fn.update_state(y_true=y_, y_pred=pred)
            loc += batch

        acc = accuracy_fn.result().numpy()
        return tf.reduce_mean(losses).numpy(), acc

    def fit(self, x, y,
            batch_size,
            epochs,
            datagen,
            validation_data,
            validation_freq,
            tsize=30000,
            shuffle=True,
            save_best_weights=True):
        """ Trains the model for a given number of epochs.
        Args:
            x - (list of arrays) - input data.
            y - (array) - target data.
            batch_size - (int) - number of samples per gradient update. 
            epochs - (int) - number of epochs to train the model. 
            datagen - (tensorflow.keras.preprocessing.image.ImageDataGenerator 
                instance) - generator that handles real-time data augmentation,
                shuffling, and batching.
            validation_data - (tuple) input and target data on which to evaluate
                the loss and accuracy. For example, (x_test, y_test).
            validation_freq - (int or None) - Frequency of validation. For value n, 
                the model is validated every n batches. If None, no validation is 
                performed.
            tsize - (int) - size of train data (x and y) used to evaluate the 
                train loss and accuracy. For example, 10000 meaning 10000 examples
                from x will be randomly selected for train evaluation.
            shuffle - (boolean) - whether to shuffle the data.
            save_best_weights - (boolean) - whether to save the weights with the
                highest validation accuracy.
        """

        self.LaPerm.sort_theta()

        self.best_weights = None
        self.best_accuracy = 0

        self.iterations = 0

        dataflow = datagen.flow(x=x, y=y,
                                batch_size=batch_size, shuffle=shuffle)
        val_x, val_y = validation_data

        epoch_bar = pbar(epochs, desc='Epoch')

        # collect evaluation data before training
        vloss, vacc, tloss, tacc = 0, 0, 0, 0
        if validation_freq:
            tindices = random.sample(range(len(x)), tsize)
            tloss, tacc = self.evaluate(x[tindices], y[tindices])
            vloss, vacc = self.evaluate(val_x, val_y)
            self._history = {'val accuracy': [vacc],
                             'val loss': [vloss],
                             'accuracy': [tacc],
                             'loss': [tloss]}

        epoch_bar = pbar(epochs, desc='Epoch', leave=True, ncols=800)
        for epoch in epoch_bar:

            batch_num = -(-len(x)//batch_size)
            batch_bar = pbar(batch_num, desc='Batch', leave=False, ncols=800)
            batch_bar.set_description(
                "vloss:{:.3f} vacc:{:.3f} tloss:{:.3f} tacc:{:.3f}"
                .format(vloss, vacc, tloss, tacc))
            batch_bar.refresh()

            for batch in batch_bar:
                local_step = self.iterations+1

                lr = self.lr_schedule(epoch)
                k = self.k_schedule(epoch)

                self.inner_optimizer.lr.assign(lr)

                # decide whether to perform synchronization
                sync_cond = (local_step//k)*k == local_step
                vali_cond = validation_freq is not None and \
                    (local_step//validation_freq)*validation_freq == local_step

                x_, y_ = next(dataflow)
                losses, grads = self.grad(x_, y_)
                self.inner_optimizer.apply_gradients(
                    zip(grads, self._model.trainable_variables))

                if sync_cond:
                    self.LaPerm.apply_synchronization(
                        self._model.trainable_variables)

                if vali_cond:
                    vloss, vacc = self.evaluate(val_x, val_y)
                    self._history['val loss'].append(vloss)
                    self._history['val accuracy'].append(vacc)

                    tindices = random.sample(range(len(x)), tsize)
                    tloss, tacc = self.evaluate(x[tindices], y[tindices])
                    self._history['loss'].append(tloss)
                    self._history['accuracy'].append(tacc)

                    batch_bar.set_description(
                        "vloss:{:.3f} vacc:{:.3f} tloss:{:.3f} tacc:{:.3f}"
                        .format(vloss, vacc, tloss, tacc))
                    batch_bar.refresh()

                    if save_best_weights and vacc > self.best_accuracy:
                        self.best_weights = self._model.get_weights()
                        self.best_accuracy = vacc

                epoch_bar.set_description(
                    "best vacc:{:.3f}".format(self.best_accuracy))
                epoch_bar.refresh()

                self.iterations = local_step
