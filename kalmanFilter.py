class KalmanFilter:
      def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
          self.process_variance = process_variance
          self.measurement_variance = measurement_variance
          self.estimated_measurement_variance = estimated_measurement_variance
          self.posteri_estimate = 0.0
          self.posteri_error_estimate = 1.0
          self.last_breathing_rate = 0

      def update(self, measurement):
          priori_estimate = self.posteri_estimate
          priori_error_estimate = self.posteri_error_estimate + self.process_variance

          blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
          self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
          self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

          return self.posteri_estimate

  