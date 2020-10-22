# # [TODO] This code is under development, use at own risk

# import numpy as np
# import tensorflow as tf

# class LaPerm(tf.keras.optimizers.Optimizer):
#     """
#     """

#     def __init__(
#         self,
#         optimizer,
#         sync_period: int = 20,
#         name: str = "LaPerm",
#         **kwargs
#     ):
#         r"""
#         """
#         super().__init__(name, **kwargs)

#         if isinstance(optimizer, str):
#             optimizer = tf.keras.optimizers.get(optimizer)
#         if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
#             raise TypeError(
#                 "optimizer is not an object of tf.keras.optimizers.Optimizer"
#             )

#         self._optimizer = optimizer
#         self._set_hyper("sync_period", sync_period)
#         self._initialized = False

#     def _create_slots(self, var_list):
#         self._optimizer._create_slots(
#             var_list=var_list
#         )  # pylint: disable=protected-access
#         for var in var_list:
#             self.add_slot(var, "sorted")

#     def _create_hypers(self):
#         self._optimizer._create_hypers()  # pylint: disable=protected-access

#     def _prepare(self, var_list):
#         return self._optimizer._prepare(
#             var_list=var_list
#         )  # pylint: disable=protected-access

#     def apply_gradients(self, grads_and_vars, name=None, **kwargs):
#         self._optimizer._iterations = (
#             self.iterations
#         )  # pylint: disable=protected-access
#         return super().apply_gradients(grads_and_vars, name, **kwargs)

#     def make_vectors_1d(self, var):
#         """ Please refer to make_vectors_4d. 
#         """
#         return tf.expand_dims(var, axis=0)

#     def make_vectors_2d(self, var):
#         """ Please refer to make_vectors_4d.
#         """
#         return tf.transpose(var)

#     def make_vectors_4d(self, var):
#         """ Transforms var into 2d matrix where each 
#         row is a weight vector. 
#         """
#         w = tf.transpose(var, perm=tf.convert_to_tensor([3, 0, 1, 2]))
#         return tf.reshape(w, shape=tf.convert_to_tensor([w.shape[0], -1]))

#     def make_variable_1d(self, var, oldshape):
#         """ Please refer to make_variable_4d. 
#         """
#         return var[0]

#     def make_variable_2d(self, var, oldshape):
#         """ Please refer to make_variable_4d. 
#         """
#         return tf.transpose(var)

#     def make_variable_4d(self, var, oldshape):
#         """ Revert the effect of make_vectors_4d(var).
#         """
#         shape = tf.convert_to_tensor(
#             [oldshape[3], oldshape[0], oldshape[1], oldshape[2]])
#         return tf.transpose(tf.reshape(var, shape=shape), perm=tf.convert_to_tensor([1, 2, 3, 0]))

#     def sort_1d(self, var):
#         return tf.sort(self.make_vectors_1d(var))

#     def sort_2d(self, var):
#         return tf.sort(self.make_vectors_2d(var))

#     def sort_4d(self, var):
#         return tf.sort(self.make_vectors_4d(var))

#     def _sort_theta(self, var):
#         """Create theta_sorted.
#         Creates a copy of theta_0 (the initial, e.g., random weights) 
#         and ascendingly sorts every weight vector of this copy.
#         The outcome is then stored as theta_sorted.
#         """
        
#         old_shape = var.shape
#         if var.shape.ndims == 1:
#             var = self.sort_1d(var)
#             var = self.make_variable_1d(var, old_shape)
#         elif var.shape.ndims  == 2:
#             var = self.sort_2d(var)
#             var = self.make_variable_2d(var, old_shape)
#         elif var.shape.ndims  == 4:
#             var = self.sort_4d(var)
#             var = self.make_variable_4d(var, old_shape)
#         else:
#             raise NotImplementedError(
#                 "Unknown tensor rank: {}".format(var.shape.ndims))
#         return var

#     def _init_op(self, var):
#         temp_var = self.get_slot(var, "sorted")
#         sorted_var = self._sort_theta(temp_var)
#         return temp_var.assign(
#             tf.where(
#                 tf.equal(self.iterations, tf.constant(0, dtype=self.iterations.dtype)),
#                 var,
#                 sorted_var,
#             ),
#             use_locking=self._use_locking,
#         )

#     def _synchronization_op(self, var):
#         var_dtype = var.dtype.base_dtype
#         sorted_var = self.get_slot(var, "sorted")
#         local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
#         sync_period = self._get_hyper("sync_period", tf.dtypes.int64)

#         permuted_var = self.apply_synchronization(var, sorted_var)
#         sync_cond = tf.equal(
#             tf.math.floordiv(local_step, sync_period) * sync_period, local_step
#         )

#         with tf.control_dependencies([permuted_var]):
#             var_update = var.assign(
#                 tf.where(sync_cond, permuted_var, var), use_locking=self._use_locking
#             )
#         return var_update
    
#     def apply_synchronization(self, theta, theta_sorted):
#         """Synchronization.
#         Permute every weight matrix theta_sorted_i in theta_sorted according to
#         its counterpart theta_t_i in theta_t.
#         """

#         # rank = tf.rank(theta_t_i)
#         theta_permuted = theta
#         if theta.shape.ndims == 1:
#             theta_permuted = self.synchronize_1d(theta, theta_sorted)
#         elif theta.shape.ndims == 2:
#             theta_permuted = self.synchronize_2d(theta, theta_sorted)
#         elif theta.shape.ndims == 4:
#             theta_permuted = self.synchronize_4d(theta, theta_sorted)
#         else:
#             raise NotImplementedError(
#                 "Unknown tensor rank: {}".format(theta.shape.ndims))
#         return theta_permuted

#     def synchronize_1d(self, var, var_sorted):
#         """ Synchronize var (if ndim(var) == 1)
#         Please refer to synchronize_4d.
#         """
#         old_shape = var.shape
#         var_shape = [1, var.shape[0]]
#         var = self.make_vectors_1d(var)
#         var_sorted = self.make_vectors_1d(var_sorted)
#         permuted = self.permute(var, var_shape, var_sorted)
#         return self.make_variable_1d(permuted, old_shape)

#     def synchronize_2d(self, var, var_sorted):
#         """ Synchronize var (if ndim(var) == 2)
#         Please refer to synchronize_4d.
#         """
#         old_shape = var.shape
#         var_shape = [var.shape[1], var.shape[0]]
#         var = self.make_vectors_2d(var)
#         var_sorted = self.make_vectors_2d(var_sorted)
#         permuted = self.permute(var, var_shape, var_sorted)
#         return self.make_variable_2d(permuted, old_shape)

#     def synchronize_4d(self, var, var_sorted):
#         """ Synchronize var (if ndim(var) == 4)
#         (1) transforms var into a 2d weight matrix where each row is 
#         an 1d weight vector. (2) produce a permuted version of var_sorted 
#         against this weight matrix obtained in (1). 
#         (3) transforms the permuted var_sorted back to the shape of var.
#         """
#         old_shape = var.shape
#         var_shape = [var.shape[3], var.shape[0]*var.shape[1]*var.shape[2]]
#         var = self.make_vectors_4d(var)
#         var_sorted = self.make_vectors_4d(var_sorted)
#         permuted = self.permute(var, var_shape, var_sorted)
#         return self.make_variable_4d(permuted, old_shape)

#     def permute(self, var, var_shape, var_sorted):
#         """ Permute var_sorted w.r.t. the rankings of var, such that
#         weight vectors w_j in var and w_j_prime in var_sorted has the 
#         same ranking for all j.

#         Args:
#             var - (2d tensor) - a tensor where each row is a 
#                 weight vector to be permuted
#             var_shape - (tensor) - the shape of var
#             var_sorted - (2d tensor) - the weight matrix to 
#                 be permuted against var.
#         """

#         ranking = tf.argsort(tf.argsort(var))
#         row_idx = tf.transpose(tf.repeat(tf.expand_dims(
#             tf.range(0, var_shape[0]), 0), [var_shape[1]], axis=0))
#         pair = tf.stack([row_idx, ranking])
#         permutation_idx = tf.transpose(
#             pair, perm=tf.convert_to_tensor([1, 2, 0]))
#         # Perform permutation by indexing
#         return tf.gather_nd(var_sorted, permutation_idx)

#     @property
#     def weights(self):
#         return self._weights + self._optimizer.weights

#     def _resource_apply_dense(self, grad, var):
#         init_op = self._init_op(var)
#         with tf.control_dependencies([init_op]):
#             train_op = self._optimizer._resource_apply_dense(
#                 grad, var
#             )  # pylint: disable=protected-access
#             with tf.control_dependencies([train_op]):
#                 look_ahead_op = self._synchronization_op(var)
#         return tf.group(init_op, train_op, look_ahead_op)

#     def _resource_apply_sparse(self, grad, var, indices):
#         init_op = self._init_op(var)
#         with tf.control_dependencies([init_op]):
#             train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
#                 grad, var, indices
#             )
#             with tf.control_dependencies([train_op]):
#                 look_ahead_op = self._synchronization_op(var)
#         return tf.group(init_op, train_op, look_ahead_op)

#     def get_config(self):
#         config = {
#             "optimizer": tf.keras.optimizers.serialize(self._optimizer),
#             "sync_period": self._serialize_hyperparameter("sync_period"),
#         }
#         base_config = super().get_config()
#         return {**base_config, **config}

#     @property
#     def learning_rate(self):
#         return self._optimizer._get_hyper("learning_rate")

#     @learning_rate.setter
#     def learning_rate(self, learning_rate):
#         self._optimizer._set_hyper("learning_rate", learning_rate)

#     @property
#     def lr(self):
#         return self.learning_rate

#     @lr.setter
#     def lr(self, lr):
#         self.learning_rate = lr

#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         optimizer = tf.keras.optimizers.deserialize(
#             config.pop("optimizer"), custom_objects=custom_objects,
#         )
#         return cls(optimizer, **config)