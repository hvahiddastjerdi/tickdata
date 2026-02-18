import os
import time
import logging
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDateEdit,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QProgressBar,
    QDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal, QThread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_load_timings.log"),
        logging.StreamHandler(),
    ],
)
timing_logger = logging.getLogger("timing")


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        module_name = func.__module__
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            timing_logger.info(
                f"{module_name}.{func_name} completed in {duration:.4f} seconds"
            )
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            timing_logger.error(
                f"{module_name}.{func_name} failed after {duration:.4f} seconds: {e}"
            )
            raise

    return wrapper


import TDR_V5 as tickdata


class DownloadProgressBar(QProgressBar):
    stopClicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.stopClicked.emit()
        super().mousePressEvent(event)


class RangeDownloadWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list, bool, str)

    def __init__(self, symbol: str, start_dt: datetime, end_dt: datetime, point: float | None = None):
        super().__init__()
        self.symbol = symbol
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.point = point

    def run(self):
        def cb(current: int, total: int, message: str):
            self.progress.emit(current, total, message)

        try:
            ticks = tickdata.download_range(
                self.symbol,
                self.start_dt,
                self.end_dt,
                point=self.point,
                progress_callback=cb,
            )
            cancelled = getattr(tickdata, "cancel_flag", False)
            self.finished.emit(ticks, cancelled, "")
        except Exception as e:
            self.finished.emit([], False, str(e))


class DownloadRangeDialog(QDialog):
    range_selected = pyqtSignal(str, QDate, QDate)

    def __init__(self, symbol: str, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.setWindowTitle("Download Data Range")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        symbol_label = QLabel(f"Symbol: {self.symbol}")
        layout.addWidget(symbol_label)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start Date:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        default_start = QDate.currentDate().addYears(-1)
        self.start_date_edit.setDate(default_start)
        start_row.addWidget(self.start_date_edit)
        layout.addLayout(start_row)

        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End Date:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        end_row.addWidget(self.end_date_edit)
        layout.addLayout(end_row)

        buttons_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Download")
        self.btn_cancel = QPushButton("Cancel")
        buttons_row.addWidget(self.btn_start)
        buttons_row.addWidget(self.btn_cancel)
        layout.addLayout(buttons_row)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel.clicked.connect(self.reject)

    def _on_start(self):
        self.range_selected.emit(
            self.symbol,
            self.start_date_edit.date(),
            self.end_date_edit.date(),
        )
        self.accept()


class TickDataDownloader(QMainWindow):
    def __init__(self, data_folder: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Center")
        self.resize(900, 600)
        self.data_folder = data_folder
        self.symbol_files = {}
        self.active_downloads = {}
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Data folder:"))
        self.edit_folder = QLineEdit(self.data_folder)
        row1.addWidget(self.edit_folder)
        self.btn_change_folder = QPushButton("Change Folder")
        row1.addWidget(self.btn_change_folder)
        top_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Data provider:"))
        self.combo_provider = QComboBox()
        self.combo_provider.addItem("Dukascopy")
        row2.addWidget(self.combo_provider)
        top_layout.addLayout(row2)

        main_layout.addWidget(top_panel)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "",
            "Symbol",
            "Downloaded Range",
            "Update",
            "Download",
            "Clear",
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        main_layout.addWidget(self.table)
        self.table.cellClicked.connect(self._on_table_cell_clicked)

        bulk_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select / Deselect All")
        self.btn_update_selected = QPushButton("Update All Selected")
        self.btn_download_selected = QPushButton("Download All Selected")
        self.btn_clear_selected = QPushButton("Clear All Selected")
        self.btn_clear_selected.setStyleSheet("color: red;")
        bulk_row.addWidget(self.btn_select_all)
        bulk_row.addWidget(self.btn_update_selected)
        bulk_row.addWidget(self.btn_download_selected)
        bulk_row.addWidget(self.btn_clear_selected)
        main_layout.addLayout(bulk_row)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.btn_close = QPushButton("Close")
        bottom_row.addWidget(self.btn_close)
        bottom_row.addStretch()
        main_layout.addLayout(bottom_row)

        self.btn_change_folder.clicked.connect(self._on_change_folder)
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_close.clicked.connect(self.close)

        self._populate_sample_rows()

    @measure_time
    def _populate_sample_rows(self):
        symbol_files = {}
        if os.path.isdir(self.data_folder):
            for name in os.listdir(self.data_folder):
                if not name.lower().endswith(".parquet"):
                    continue
                symbol = name.split("_")[0]
                full_path = os.path.join(self.data_folder, name)
                symbol_files.setdefault(symbol, []).append(full_path)

        self.symbol_files = symbol_files

        symbols = sorted(symbol_files.keys())
        self.table.setRowCount(len(symbols))

        for row, symbol in enumerate(symbols):
            item_check = QTableWidgetItem()
            item_check.setFlags(item_check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_check.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, item_check)

            item_symbol = QTableWidgetItem(symbol)
            self.table.setItem(row, 1, item_symbol)

            start_ts, end_ts = self._get_symbol_range(symbol_files.get(symbol, []))
            if start_ts and end_ts:
                range_text = f"{start_ts} → {end_ts}"
            else:
                range_text = "Not downloaded"
            item_range = QTableWidgetItem(range_text)
            self.table.setItem(row, 2, item_range)

            btn_update = QPushButton("Update")
            btn_update.setFlat(True)
            btn_update.setStyleSheet("color: #2980b9; text-decoration: underline;")
            self.table.setCellWidget(row, 3, btn_update)

            btn_download = QPushButton("Download")
            self.table.setCellWidget(row, 4, btn_download)

            btn_clear = QPushButton("Clear")
            btn_clear.setStyleSheet("color: red;")
            self.table.setCellWidget(row, 5, btn_clear)

            btn_download.clicked.connect(lambda _, r=row: self._on_row_download(r))
            btn_update.clicked.connect(lambda _, r=row: self._on_row_update(r))

    def _on_change_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", self.data_folder)
        if folder:
            self.data_folder = folder
            self.edit_folder.setText(folder)
            self._populate_sample_rows()

    def _on_select_all(self):
        all_checked = True
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.checkState() != Qt.CheckState.Checked:
                all_checked = False
                break
        new_state = Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(new_state)

    def _on_table_cell_clicked(self, row: int, column: int):
        if column == 4 and row in self.active_downloads:
            setattr(tickdata, "cancel_flag", True)

    def _on_row_download(self, row: int):
        symbol_item = self.table.item(row, 1)
        if not symbol_item:
            return
        symbol = symbol_item.text()
        dialog = DownloadRangeDialog(symbol, self)

        def handle_range(sym, start, end):
            start_dt = datetime(start.year(), start.month(), start.day())
            end_dt = datetime(end.year(), end.month(), end.day()) + timedelta(days=1)
            self._start_row_download(row, sym, start_dt, end_dt, append_path=None)

        dialog.range_selected.connect(handle_range)
        dialog.exec()

    @measure_time
    def _on_row_update(self, row: int):
        symbol_item = self.table.item(row, 1)
        if not symbol_item:
            return
        symbol = symbol_item.text()
        files = self.symbol_files.get(symbol, [])
        if not files:
            return
        # Determine last date from existing files
        last_date = None
        last_file = None
        for path in files:
            _, file_end = self._get_file_range(path)
            if not file_end:
                continue
            try:
                d = datetime.strptime(file_end, "%Y-%m-%d").date()
            except ValueError:
                continue
            if last_date is None or d > last_date:
                last_date = d
                last_file = path
        if last_date is None or last_file is None:
            return
        start_date = last_date + timedelta(days=1)
        today = datetime.utcnow().date()
        if start_date > today:
            return
        start_dt = datetime(start_date.year, start_date.month, start_date.day)
        end_dt = datetime(today.year, today.month, today.day) + timedelta(days=1)
        self._start_row_download(row, symbol, start_dt, end_dt, append_path=last_file)

    @measure_time
    def _start_row_download(self, row: int, symbol: str, start_dt: datetime, end_dt: datetime, append_path: str | None):
        if row in self.active_downloads:
            return

        worker = RangeDownloadWorker(symbol, start_dt, end_dt)
        progress = DownloadProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFormat("Stop (0%)")

        self.table.setCellWidget(row, 4, progress)

        self.active_downloads[row] = {
            "worker": worker,
            "progress": progress,
            "append_path": append_path,
            "symbol": symbol,
        }

        def on_progress(current: int, total: int, msg: str):
            if total <= 0:
                return
            percent = int((current / total) * 100)
            progress.setValue(percent)
            progress.setFormat(f"Stop ({percent}%)")

        def on_finished(ticks: list, cancelled: bool, error: str):
            info = self.active_downloads.pop(row, None)
            if error:
                btn = QPushButton("Download")
                btn.clicked.connect(lambda _, r=row: self._on_row_download(r))
                self.table.setCellWidget(row, 4, btn)
                QMessageBox.critical(self, "Download Error", error)
                return

            path = append_path
            if path is None:
                start_str = start_dt.strftime("%Y%m%d")
                end_str = (end_dt - timedelta(days=1)).strftime("%Y%m%d")
                filename = f"{symbol}_ticks_{start_str}_{end_str}.parquet"
                path = os.path.join(self.data_folder, filename)

            if ticks:
                self._write_ticks_to_parquet(path, ticks, append=append_path is not None)

            # Refresh table ranges
            self._populate_sample_rows()

        def on_stop_clicked():
            tickdata.cancel_download()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        progress.stopClicked.connect(on_stop_clicked)

        worker.start()

    def _write_ticks_to_parquet(self, filepath: str, ticks: list, append: bool):
        if not ticks:
            return
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(ticks)
        
        # Ensure required columns exist
        required_cols = ['datetime', 'ask', 'bid', 'ask_volume', 'bid_volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Parse datetime if needed
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Convert to numpy structured array for TDR_V5
        import numpy as np
        from datetime import timezone
        
        n = len(df)
        arr = np.empty(n, dtype=[
            ('timestamp_us', 'i8'),
            ('ask', 'f8'),
            ('bid', 'f8'),
            ('ask_volume', 'f4'),
            ('bid_volume', 'f4'),
        ])
        
        # Convert datetime to microseconds
        if 'datetime' in df.columns:
            arr['timestamp_us'] = df['datetime'].astype(np.int64) // 1000
        else:
            arr['timestamp_us'] = np.arange(n, dtype=np.int64)
        
        arr['ask'] = df.get('ask', 0.0).astype(np.float64)
        arr['bid'] = df.get('bid', 0.0).astype(np.float64)
        arr['ask_volume'] = df.get('ask_volume', 0.0).astype(np.float32)
        arr['bid_volume'] = df.get('bid_volume', 0.0).astype(np.float32)
        
        # Use TDR_V5 to save
        if append and os.path.exists(filepath):
            # Load existing data and append
            try:
                existing_ticks = tickdata.load_ticks(filepath)
                combined_ticks = np.concatenate([existing_ticks, arr])
                tickdata.save_ticks(combined_ticks, filepath)
            except Exception as e:
                # If append fails, just save the new data
                tickdata.save_ticks(arr, filepath)
        else:
            tickdata.save_ticks(arr, filepath)

    @measure_time
    def _get_symbol_range(self, files):
        if not files:
            return None, None
        start_ts = None
        end_ts = None
        for path in files:
            file_start, file_end = self._get_file_range(path)
            if not file_start or not file_end:
                continue
            if start_ts is None or file_start < start_ts:
                start_ts = file_start
            if end_ts is None or file_end > end_ts:
                end_ts = file_end
        return start_ts, end_ts

    @measure_time
    def _get_file_range(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline()
                if not header:
                    return None, None
                cols = header.strip().split(",")
                try:
                    dt_idx = cols.index("datetime")
                except ValueError:
                    dt_idx = len(cols) - 1
                first_data = f.readline()
                if not first_data:
                    return None, None
                first_parts = first_data.strip().split(",")
                if len(first_parts) <= dt_idx:
                    return None, None
                first_full = first_parts[dt_idx]
                first_ts = first_full.split(" ")[0]

                last_line = first_data
                for line in f:
                    if line.strip():
                        last_line = line
                last_parts = last_line.strip().split(",")
                if len(last_parts) <= dt_idx:
                    return None, None
                last_full = last_parts[dt_idx]
                last_ts = last_full.split(" ")[0]
                return first_ts, last_ts
        except OSError:
            return None, None
