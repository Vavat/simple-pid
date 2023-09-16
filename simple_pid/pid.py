def _clamp(value, limits):
    lower, upper = limits
    if (upper is not None) and (value > upper):
        value = upper
    elif (lower is not None) and (value < lower):
        value = lower
    return value

class PID(object):
    """A simple PID controller."""

    def __init__(
        self,
        k_proportional=1.0,
        k_integral=0.0,
        k_differential=0.0,
        setpoint=0,
        sample_time=0.01,
        output_limits=(None, None),
        auto_mode=True,
        proportional_on_measurement=False,
        differential_on_measurement=True,
        error_map=None,
        time_fn=None,
        integral_preload=0.0,
    ):
        """
        Initialize a new PID controller.

        :param k_proportional: The value for the proportional gain k_proportional
        :param  k_integral: The value for the integral gain k_integral
        :param k_differential: The value for the derivative gain k_differential
        :param setpoint: The initial setpoint that the PID will try to achieve
        :param sample_time: The time in seconds which the controller should wait before generating
            a new output value. The PID works best when it is constantly called (eg. during a
            loop), but with a sample time set so that the time difference between each update is
            (close to) constant. If set to None, the PID will compute a new output value every time
            it is called.
        :param output_limits: The initial output limits to use, given as an iterable with 2
            elements, for example: (lower, upper). The output will never go below the lower limit
            or above the upper limit. Either of the limits can also be set to None to have no limit
            in that direction. Setting output limits also avoids integral windup, since the
            integral term will never be allowed to grow outside of the limits.
        :param auto_mode: Whether the controller should be enabled (auto mode) or not (manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on
            the input directly rather than on the error (which is the traditional way). Using
            proportional-on-measurement avoids overshoot for some types of systems.
        :param differential_on_measurement: Whether the differential term should be calculated on
            the input directly rather than on the error (which is the traditional way).
        :param error_map: Function to transform the error value in another constrained value.
        :param time_fn: The function to use for getting the current time, or None to use the
            default. This should be a function taking no arguments and returning a number
            representing the current time. The default is to use time.monotonic() if available,
            otherwise time.time().
        :param integral_preload: The starting point for the PID's output. If you start controlling
            a system that is already at the setpoint, you can set this to your best guess at what
            output the PID should give when first calling it to avoid the PID outputting zero and
            moving the system away from the setpoint.
        """
        self.k_proportional, self.k_integral, self.k_differential = k_proportional, k_integral, k_differential
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        self.p_term = 0
        self.i_term = 0
        self.d_term = 0

        self._last_time = None
        self._last_output = None
        self._last_error = None
        self._last_input = None

        if time_fn is not None:
            # Use the user supplied time function
            self.time_fn = time_fn
        else:
            import time

            try:
                # Get monotonic time to ensure that time deltas are always positive
                self.time_fn = time.monotonic
            except AttributeError:
                # time.monotonic() not available (using python < 3.3), fallback to time.time()
                self.time_fn = time.time

        self.output_limits = output_limits
        self.reset()

        # Set initial state of the controller
        self.i_term = _clamp(integral_preload, output_limits)

    def __call__(self, controlled_value, delta_t=None):
        """
        Update the PID controller.

        Call the PID controller with *controlled_value* and calculate and return a control output if
        sample_time seconds has passed since the last update. If no new output is calculated,
        return the previous output instead (or None if no value has been calculated yet).

        :param delta_t: If set, uses this value for timestep instead of real time. This can be used in
            simulations when simulation time is different from real time.
        """
        if not self.auto_mode:
            return self._last_output

        now = self.time_fn()
        if delta_t is None:
            delta_t = now - self._last_time if (now - self._last_time) else 1e-16
        elif delta_t <= 0:
            raise ValueError('delta_t has negative value {}, must be positive'.format(delta_t))

        # Compute error terms
        error = self.setpoint - controlled_value
        delta_error = error - self._last_error # N.B.: not time normalised

        # Compute terms: proportional and differential
        self.p_term = self.k_proportional * error 
        old_i_term = self.i_term
        self.i_term += self.k_integral * error * delta_t
        self.d_term = self.k_differential* delta_error / delta_t

        output = self.p_term + self.i_term + self.d_term

        # Limit checking of the output and windup prevention
        minimum_output, maximum_output = self.output_limits
        if output < minimum_output:
            output = minimum_output
            # if lower limit is hit, then prevent integral from reducing further
            if self.i_term < old_i_term:
                self.i_term = old_i_term
        elif output > maximum_output:
            output = maximum_output
            # if upper limit is hit, then prevent integral from increasing further
            if self.i_term > old_i_term:
                self.i_term = old_i_term
        
        # Keep track of state
        self._last_output = output
        self._last_input = controlled_value
        self._last_error = error
        self._last_time = now

        return output

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'k_proportional={self.k_proportional!r}, k_integral={self.k_integral!r}, k_differential={self.k_differential!r}, '
            'setpoint={self.setpoint!r}, sample_time={self.sample_time!r}, '
            'output_limits={self.output_limits!r}, auto_mode={self.auto_mode!r}, '
            'proportional_on_measurement={self.proportional_on_measurement!r}, '
            'differential_on_measurement={self.differential_on_measurement!r}, '
            'error_map={self.error_map!r}'
            ')'
        ).format(self=self)

    @property
    def components(self):
        """
        The P-, I- and D-terms from the last computation as separate components as a tuple. Useful
        for visualizing what the controller is doing or when tuning hard-to-tune systems.
        """
        return self.p_term, self.i_term, self.d_term

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (k_proportional,  k_integral, k_differential)."""
        return self.k_proportional, self.k_integral, self.k_differential

    @tunings.setter
    def tunings(self, tunings):
        """Set the PID tunings."""
        self.k_proportional, self.k_integral, self.k_differential = tunings

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not."""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PID controller."""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        Enable or disable the PID controller, optionally setting the last output value.

        This is useful if some system has been manually controlled and if the PID should take over.
        In that case, disable the PID by setting auto mode to False and later when the PID should
        be turned back on, pass the last output variable (the control variable) and it will be set
        as the starting I-term when the PID is set to auto mode.

        :param enabled: Whether auto mode should be enabled, True or False
        :param last_output: The last output, or the control variable, that the PID should start
            from when going from manual mode to auto mode. Has no effect if the PID is already in
            auto mode.
        """
        if enabled and not self._auto_mode:
            # Switching from manual mode to auto, reset
            self.reset()

            self.i_term = last_output if (last_output is not None) else 0
            self.i_term = _clamp(self.i_term, self.output_limits)

        self._auto_mode = enabled

    @property
    def output_limits(self):
        """
        The current output limits as a 2-tuple: (lower, upper).

        See also the *output_limits* parameter in :meth:`PID.__init__`.
        """
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Set the output limits."""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        if (None not in limits) and (max_output < min_output):
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        self.i_term = _clamp(self.i_term, self.output_limits)
        self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        """
        Reset the PID controller internals.

        This sets each term to 0 as well as clearing the integral, the last output and the last
        input (derivative calculation).
        """
        self.p_term = 0
        self.i_term = 0
        self.d_term = 0

        self.i_term = _clamp(self.i_term, self.output_limits)

        self._last_time = self.time_fn()
        self._last_output = None
        self._last_input = None
