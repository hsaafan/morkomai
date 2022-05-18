import subprocess


class Display:
    def __init__(self, display_id: int = 13) -> None:
        """Display class used to create a display server using Xephyr.

        Parameters
        ----------
        display_id: int, optional
            An id used to identify the display, defaults to 13.

        Attributes
        ----------
        display_id: int
            An id used to identify the display.
        is_running: bool
            True if display server is still running.
        """
        self._display_id = display_id
        self._process = None
        self._preface = f'DISPLAY=:{self.display_id} '

    # Properties
    def _get_display_id(self) -> int: return(self._display_id)
    display_id = property(fget=_get_display_id,
                          doc="The id of the display server.")

    def _get_status(self) -> bool:
        if self._process is None:
            return(False)
        else:
            return(self._process.poll() is None)
    is_running = property(fget=_get_status,
                          doc="True if display server is running.")

    # Methods
    def start(self) -> None:
        """Starts a display server using Xephyr."""
        cmd = f'Xephyr :{self.display_id} -screen 640x480'
        self._process = subprocess.Popen(cmd, shell=True)

    def check_still_running(self) -> None:
        """Checks that the display server is running."""
        if not self.is_running:
            raise ChildProcessError('The display server has stopped')

    def call(self, command: str) -> None:
        """Run commands on display server.

        Parameters
        ----------
        command: str
            The command to run.
        """
        subprocess.call(self._preface + command, shell=True)

    def popen(self, command: str) -> subprocess.Popen:
        """Open process in display server.

        Parameters
        ----------
        command: str
            The command to run.

        Returns
        -------
        process: subprocess.Popen
            The process object.
        """
        process = subprocess.Popen(self._preface + command, shell=True)
        return(process)

    def check_output(self, command: str) -> str:
        """Run commands on display server and returns output.

        Command can raise a subprocess.CalledProcessError for commands that
        return with a non-zero exit status.

        Parameters
        ----------
        command: str
            The command to run.

        Returns
        -------
        output: str
            A byte string that contains the output of the command.
        """
        output = subprocess.check_output(self._preface + command, shell=True)
        return(output)

    def keystroke(self, key: str, time_held: float = 50) -> None:
        """Send keystroke to display server using xdotool key.

        Parameters
        ----------
        key: str
            The keystroke to send, consult xdotool for the list of keys.
        time_held: float, optional
            The time in ms to hold the key down, the default is 50.
        """
        self.check_still_running()
        self.call(f'xdotool key --delay {time_held} "{key}"')

    def send_string(self, string: str, press_enter: bool = False) -> None:
        """Send typed string to display server using xdotool type command.

        Parameters
        ----------
        string: str
            The string to be typed. Double quotes " in string are replaced with
            single quotes '.
        press_enter: bool, optional
            If true, a "Return" keystroke will be sent after the string is
            typed.
        """
        self.check_still_running()
        string = string.replace('"', "'")
        self.call(f'xdotool type {string}')
        if press_enter:
            self.keystroke('Return')

    def keydown(self, key: str) -> None:
        """Press key down on display server using xdotool keydown command.

        The key remains pressed down until the keyup command is called on the
        same key.

        Parameters
        ----------
        key: str
            The key to press down, consult xdotool for the list of keys.
        """
        self.check_still_running()
        self.call(f'xdotool keydown {key}')

    def keyup(self, key: str) -> None:
        """Release key on display server using xdotool keyup command.

        Keys can be pressed down using keydown command.

        Parameters
        ----------
        key: str
            The key to press down, consult xdotool for the list of keys.
        """
        self.check_still_running()
        self.call(f'xdotool keyup {key}')

    def stop(self) -> None:
        """Terminate the display server process."""
        self._process.terminate()
