import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .logging_setup import install_excepthook, setup_logging
from .ui.main_window import MainWindow

log = logging.getLogger("fourier_atlas")


def main() -> int:
    log_path = setup_logging()
    install_excepthook()
    app = QApplication(sys.argv)
    app.setApplicationName("Fourier Atlas")
    win = MainWindow()
    install_excepthook(lambda msg: QMessageBox.critical(
        win, "Непредвиденная ошибка", f"{msg}\n\nПодробности записаны в журнал:\n{log_path}"))
    win.show()
    log.info("Fourier Atlas запущен")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
