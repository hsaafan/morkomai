import subprocess
import time
from display import Display
import psutil


class DOSBox:
    def __init__(self, display: Display = None) -> None:
        """Class used to run dosbox application.

        Parameters
        ----------
        display: Display, optional
            The display server object to run dosbox on. If none are passed, a
            new display server will be started with the default display id.

        Attributes
        ----------
        pid: str
            The pid of the dosbox GUI.
        is_running: bool
            True if dosbox application is still running.
        """
        if display is None:
            display = Display()
        self._shell = None
        self._display = display
        self._pid = None
        self._inherit_keys()

    def _get_pid(self) -> str: return(self._pid)
    pid = property(fget=_get_pid, doc='The pid of the dosbox GUI.')

    def _get_status(self) -> bool:
        if self._shell is None:
            return(False)
        else:
            shell_running = self._shell.poll() is None
            gui_running = psutil.pid_exists(self.pid)
            return(shell_running and gui_running)
    is_running = property(fget=_get_status,
                          doc="True if dosbox application is running.")

    def check_still_running(self) -> None:
        """Checks that dosbox is running."""
        if not self.is_running:
            raise ChildProcessError('dosbox has stopped')

    def start(self, mount_folder: str = None, conf_file: str = None) -> None:
        """Start the dosbox program.

        Starts dosbox and grabs the pid and window id of the GUI.

        Parameters
        ----------
        mount_folder: str, optional
            Mount folder for dosbox. Mounting happens after autoexec of conf
            file. Mount commands can be added to conf file to mount before
            other autoexec commands.
        conf_file: str, optional
            dosbox conf file to use.
        """
        if not self._display.is_running:
            self._display.start()
            time.sleep(0.25)
        cmd = "dosbox"
        if mount_folder is not None:
            cmd += f" {mount_folder}"
        if conf_file is not None:
            cmd += f' -conf {conf_file}'
        self._shell = self._display.popen(cmd)
        self._store_GUI_PID()

    def _store_GUI_PID(self) -> None:
        """Get the PID of dosbox GUI by checking children of dosbox shell."""
        while True:
            try:
                pid = self._display.check_output(f'pgrep -P {self._shell.pid}')
                break
            except subprocess.CalledProcessError:
                time.sleep(0.25)
        self._pid = pid.decode('utf-8').strip()

    def stop(self) -> None:
        """Kill the dosbox GUI."""
        self._display.call(f'kill -9 {self.pid}')

    def _inherit_keys(self):
        """Inherit the key methods from the display server"""
        self.keystroke = self._display.keystroke
        self.send_string = self._display.send_string
        self.keydown = self._display.keydown
        self.keyup = self._display.keyup
        self.capture = self._display.capture

    def fast_forward(self):
        self.keydown('ALT+F12')
