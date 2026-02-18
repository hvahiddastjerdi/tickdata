import sys
import os
import time as time_module
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QDateEdit, QPushButton,
    QProgressBar, QStatusBar, QGroupBox, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
import pyqtgraph as pg
import finplot as fplt
import requests as req

# Setup logging for timing measurements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_load_timings.log'),
        logging.StreamHandler()
    ]
)
timing_logger = logging.getLogger('timing')

# ============================================================
# TIMING DECORATOR
# ============================================================

def measure_time(func):
    """Decorator to measure function execution time and log results."""
    def wrapper(*args, **kwargs):
        start_time = time_module.time()
        func_name = func.__name__
        module_name = func.__module__
        
        try:
            result = func(*args, **kwargs)
            end_time = time_module.time()
            duration = end_time - start_time
            
            timing_logger.info(f"{module_name}.{func_name} completed in {duration:.4f} seconds")
            return result
        except Exception as e:
            end_time = time_module.time()
            duration = end_time - start_time
            
            timing_logger.error(f"{module_name}.{func_name} failed after {duration:.4f} seconds: {str(e)}")
            raise
    
    return wrapper

# Import data logic
import TDR_V5 as tickdata
from tickdatadownloader import TickDataDownloader

# ============================================================
# DOWNLOAD WORKER THREAD
# ============================================================
class DownloadWorker(QThread):
    """Background thread for downloading tick data."""
    progress = pyqtSignal(int, int, str)   # current, total, message
    finished = pyqtSignal(list)             # ticks
    error = pyqtSignal(str)                 # error message

    def __init__(self, symbol, start_date, end_date, point):
        super().__init__()
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.point = point
        self._abort = False

    def abort(self):
        self._abort = True
        tickdata.cancel_download()

    def run(self):
        try:
            session = req.Session()
            ticks = []

            current = self.start_date.replace(
                minute=0, second=0, microsecond=0
            )
            end = self.end_date.replace(
                minute=0, second=0, microsecond=0
            )

            total_hours = int(
                (end - current).total_seconds() // 3600
            ) + 1
            done = 0

            while current <= end and not self._abort:
                hour_ticks = tickdata.download_hour(
                    self.symbol, current, self.point, session
                )
                ticks.extend(hour_ticks)
                done += 1

                self.progress.emit(
                    done, total_hours,
                    f"Downloading {self.symbol} | "
                    f"{current.strftime('%Y-%m-%d %H:%M')} | "
                    f"Ticks: {len(ticks):,}"
                )
                current += timedelta(hours=1)

            session.close()

            if self._abort:
                self.progress.emit(
                    done, total_hours, "⚠️ Download aborted by user"
                )

            self.finished.emit(ticks)

        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# MAIN WINDOW
# ============================================================
class TickDataViewer(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.ticks = []
        self.candles_df = None
        self.worker = None
        self.last_save_dir = os.getcwd()
        self.data_center_window = None

        self.ax_candle = None
        self.ax_volume = None

        self._init_ui()
        self._apply_styles()

    def _init_ui(self):
        """Build the entire user interface."""
        self.setWindowTitle("📊 Tick Data Viewer — Dukascopy")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── TOP: Settings Panel ──
        settings_group = QGroupBox("⚙️  Download Settings")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(8)

        settings_layout.addWidget(QLabel("Category:"), 0, 0)
        self.combo_category = QComboBox()
        self.combo_category.addItems(tickdata.SYMBOLS.keys())
        self.combo_category.currentTextChanged.connect(self._on_category_changed)
        settings_layout.addWidget(self.combo_category, 0, 1)

        settings_layout.addWidget(QLabel("Symbol:"), 0, 2)
        self.combo_symbol = QComboBox()
        self.combo_symbol.setMinimumWidth(140)
        self.combo_symbol.currentTextChanged.connect(self._on_symbol_changed)
        settings_layout.addWidget(self.combo_symbol, 0, 3)

        settings_layout.addWidget(QLabel("Point:"), 0, 4)
        self.lbl_point = QLabel("0.00001")
        self.lbl_point.setStyleSheet("font-weight: bold; color: #00bcd4; font-size: 13px;")
        settings_layout.addWidget(self.lbl_point, 0, 5)

        settings_layout.addWidget(QLabel("Start Date:"), 1, 0)
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setDate(QDate(2025, 1, 6))
        settings_layout.addWidget(self.date_start, 1, 1)

        settings_layout.addWidget(QLabel("End Date:"), 1, 2)
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDate(QDate(2025, 1, 10))
        settings_layout.addWidget(self.date_end, 1, 3)

        settings_layout.addWidget(QLabel("Days:"), 1, 4)
        self.lbl_days = QLabel("4")
        self.lbl_days.setStyleSheet("font-weight: bold; color: #ff9800;")
        settings_layout.addWidget(self.lbl_days, 1, 5)
        self.date_start.dateChanged.connect(self._update_days_label)
        self.date_end.dateChanged.connect(self._update_days_label)

        settings_layout.addWidget(QLabel("Timeframe:"), 2, 0)
        self.combo_tf = QComboBox()
        self.combo_tf.addItems(tickdata.TIMEFRAMES)
        self.combo_tf.setCurrentText("5m")
        self.combo_tf.currentTextChanged.connect(self._on_show_chart)
        settings_layout.addWidget(self.combo_tf, 2, 1)

        settings_layout.addWidget(QLabel("Session:"), 2, 2)
        self.combo_session = QComboBox()
        self.combo_session.addItems(tickdata.SESSIONS.keys())
        self.combo_session.currentTextChanged.connect(self._on_show_chart)
        settings_layout.addWidget(self.combo_session, 2, 3)

        settings_layout.addWidget(QLabel("Price:"), 2, 4)
        self.combo_price = QComboBox()
        self.combo_price.addItems(tickdata.PRICE_TYPES.keys())
        self.combo_price.currentTextChanged.connect(self._on_show_chart)
        settings_layout.addWidget(self.combo_price, 2, 5)

        settings_layout.addWidget(QLabel("Timezone:"), 3, 0)
        self.combo_tz = QComboBox()
        self.combo_tz.addItems(tickdata.TIMEZONES.keys())
        self.combo_tz.setCurrentText("UTC")
        self.combo_tz.currentTextChanged.connect(self._on_show_chart)
        settings_layout.addWidget(self.combo_tz, 3, 1)

        btn_layout = QHBoxLayout()
        self.btn_download = QPushButton("📥  Download Ticks")
        self.btn_download.clicked.connect(lambda: self._on_download())
        btn_layout.addWidget(self.btn_download)

        self.btn_abort = QPushButton("⛔  Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._on_abort)
        btn_layout.addWidget(self.btn_abort)

        self.btn_show = QPushButton("📊  Show Chart")
        self.btn_show.setEnabled(False)
        self.btn_show.clicked.connect(self._on_show_chart)
        btn_layout.addWidget(self.btn_show)

        self.btn_qc = QPushButton("📈  QuantConnect")
        self.btn_qc.setEnabled(False)
        self.btn_qc.clicked.connect(lambda: self._on_show_quantconnect())
        btn_layout.addWidget(self.btn_qc)

        self.btn_save = QPushButton("💾  Save CSV")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save_csv)
        btn_layout.addWidget(self.btn_save)

        self.btn_load = QPushButton("📂  Load CSV")
        self.btn_load.clicked.connect(lambda: self._on_load_csv())
        btn_layout.addWidget(self.btn_load)

        self.btn_download_manager = QPushButton("📁  Download Manager")
        self.btn_download_manager.clicked.connect(self._on_open_download_manager)
        btn_layout.addWidget(self.btn_download_manager)

        settings_layout.addLayout(btn_layout, 4, 0, 1, 6)
        main_layout.addWidget(settings_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        chart_group = QGroupBox("📈  Chart")
        self.chart_layout = QVBoxLayout(chart_group)
        self.lbl_chart_info = QLabel("No data loaded")
        self.chart_layout.addWidget(self.lbl_chart_info)

        # Create a container widget for the finplot widget
        self.chart_widget_container = QWidget()
        self.chart_widget_layout = QVBoxLayout(self.chart_widget_container)
        self.chart_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_layout.addWidget(self.chart_widget_container)

        self._apply_finplot_theme()
        # Initial plot creation
        self.ax_candle = fplt.create_plot(init_zoom_periods=100)
        self.chart_widget_layout.addWidget(self.ax_candle.vb.win)

        main_layout.addWidget(chart_group, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status_ticks = QLabel("Ticks: 0")
        self.lbl_status_candles = QLabel("Candles: 0")
        self.lbl_status_symbol = QLabel("Symbol: —")
        self.lbl_status_tf = QLabel("TF: —")

        for lbl in [self.lbl_status_symbol, self.lbl_status_ticks, self.lbl_status_candles, self.lbl_status_tf]:
            lbl.setStyleSheet("padding: 0 12px; font-size: 12px;")
            self.status_bar.addPermanentWidget(lbl)

        self._on_category_changed(self.combo_category.currentText())

    def _apply_styles(self):
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
            QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 12px; padding-top: 18px; font-weight: bold; color: #89b4fa; }
            QComboBox, QDateEdit { background-color: #313244; border: 1px solid #45475a; border-radius: 5px; padding: 5px; color: #cdd6f4; }
            QPushButton { background-color: #45475a; border-radius: 6px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #585b70; }
            QProgressBar { background-color: #313244; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background: #89b4fa; }
        """)

    def _apply_finplot_theme(self):
        fplt.foreground = '#cdd6f4'
        fplt.background = '#1e1e2e'
        fplt.candle_bull_color = '#a6e3a1'
        fplt.candle_bear_color = '#f38ba8'

    def _on_category_changed(self, category):
        self.combo_symbol.blockSignals(True)
        self.combo_symbol.clear()
        if category in tickdata.SYMBOLS:
            self.combo_symbol.addItems(tickdata.SYMBOLS[category].keys())
        self.combo_symbol.blockSignals(False)
        if self.combo_symbol.count() > 0:
            self._on_symbol_changed(self.combo_symbol.currentText())

    def _on_symbol_changed(self, symbol):
        category = self.combo_category.currentText()
        if category in tickdata.SYMBOLS and symbol in tickdata.SYMBOLS[category]:
            point = tickdata.SYMBOLS[category][symbol]
            self.lbl_point.setText(f"{point}")

    def _update_days_label(self):
        start = self.date_start.date().toPyDate()
        end = self.date_end.date().toPyDate()
        days = (end - start).days
        self.lbl_days.setText(str(max(days, 0)))

    @measure_time
    def _on_download(self):
        symbol = self.combo_symbol.currentText()
        start = datetime.combine(self.date_start.date().toPyDate(), datetime.min.time())
        end = datetime.combine(self.date_end.date().toPyDate(), datetime.max.time().replace(microsecond=0))
        
        if start >= end:
            QMessageBox.warning(self, "Warning", "Start date must be before end date!")
            return

        point = float(self.lbl_point.text())
        self.btn_download.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.progress_bar.setValue(0)
        self.ticks = []
        tickdata.reset_cancel_flag()

        self.worker = DownloadWorker(symbol, start, end, point)
        self.worker.progress.connect(self._on_download_progress)
        self.worker.finished.connect(self._on_download_finished)
        self.worker.error.connect(self._on_download_error)
        self.worker.start()

    def _on_abort(self):
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.btn_abort.setEnabled(False)

    def _on_download_progress(self, current, total, message):
        percent = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(message)

    def _on_download_finished(self, ticks):
        self.ticks = ticks
        self.btn_download.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.btn_show.setEnabled(len(ticks) > 0)
        self.btn_save.setEnabled(len(ticks) > 0)
        self.btn_qc.setEnabled(len(ticks) > 0)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(f"✅ Download complete — {len(ticks):,}")
        if ticks: self._on_show_chart()

    def _on_download_error(self, error_msg):
        self.btn_download.setEnabled(True)
        self.btn_abort.setEnabled(False)
        QMessageBox.critical(self, "Error", error_msg)

    def _set_ui_enabled(self, enabled):
        """Enable/disable UI controls during processing to prevent hanging."""
        self.combo_category.setEnabled(enabled)
        self.combo_symbol.setEnabled(enabled)
        self.date_start.setEnabled(enabled)
        self.date_end.setEnabled(enabled)
        self.combo_tf.setEnabled(enabled)
        self.combo_session.setEnabled(enabled)
        self.combo_price.setEnabled(enabled)
        self.combo_tz.setEnabled(enabled)
        worker_running = bool(getattr(self, "worker", None) and self.worker.isRunning())
        self.btn_download.setEnabled(enabled and not worker_running)
        self.btn_abort.setEnabled(enabled and worker_running)
        self.btn_show.setEnabled(enabled and len(self.ticks) > 0)
        self.btn_save.setEnabled(enabled and len(self.ticks) > 0)
        self.btn_load.setEnabled(enabled)
        self.btn_download_manager.setEnabled(enabled)
        self.btn_qc.setEnabled(enabled and len(self.ticks) > 0)
        
        # Update cursor to show processing state
        if not enabled:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
            
        # Process events to update UI immediately
        QApplication.processEvents()

    def _draw_sessions(self, df, selected_session_key=None, tz_key="UTC"):
        """Draw background colors for major trading sessions using finplot's coordinate system."""
        if df is None or df.empty: return
        
        # Session times in UTC
        all_sessions = [
            ("sydney", 21, 6, "#34495e"),   # 21:00 - 06:00 UTC
            ("tokyo", 0, 9, "#16a085"),    # 00:00 - 09:00 UTC
            ("london", 7, 16, "#2980b9"),   # 07:00 - 16:00 UTC
            ("newyork", 13, 22, "#8e44ad")  # 13:00 - 22:00 UTC
        ]
        
        # Filter sessions to draw
        sessions_to_draw = []
        if selected_session_key:
            sessions_to_draw = [s for s in all_sessions if s[0] == selected_session_key]
        else:
            sessions_to_draw = all_sessions

        # Get timezone offset to adjust session rectangles on the chart
        target_tz = tickdata.get_timezone(tz_key)
        
        start_dt_data = df.index[0]
        end_dt_data = df.index[-1]
        
        current_day = start_dt_data.date() - timedelta(days=1)
        end_day = end_dt_data.date() + timedelta(days=1)
        
        # Create a list of timestamps for binary search (these are already naive/local in the DF)
        timestamps = df.index.view(np.int64) // 10**9
        
        while current_day <= end_day:
            for name, start_h, end_h, color in sessions_to_draw:
                # 1. Create session start/end in UTC
                s_utc = datetime.combine(current_day, time(hour=start_h), tzinfo=timezone.utc)
                if end_h < start_h:
                    e_utc = datetime.combine(current_day + timedelta(days=1), time(hour=end_h), tzinfo=timezone.utc)
                else:
                    e_utc = datetime.combine(current_day, time(hour=end_h), tzinfo=timezone.utc)
                
                # 2. Convert session times to the SAME timezone as the chart labels
                s_local = s_utc.astimezone(target_tz).replace(tzinfo=None)
                e_local = e_utc.astimezone(target_tz).replace(tzinfo=None)

                if e_local < start_dt_data or s_local > end_dt_data:
                    continue
                
                # 3. Find positions in the local-time index
                s_ts = s_local.timestamp()
                e_ts = e_local.timestamp()
                
                idx_start = np.searchsorted(timestamps, s_ts)
                idx_end = np.searchsorted(timestamps, e_ts)
                
                if idx_start < len(df) and idx_end > 0 and idx_start < idx_end:
                    lr = pg.LinearRegionItem(
                        values=[idx_start, idx_end],
                        orientation='vertical',
                        brush=pg.mkBrush(color+"33"),
                        movable=False
                    )
                    for line in lr.lines:
                        line.setPen(pg.mkPen(None))
                    self.ax_candle.addItem(lr)
                
            current_day += timedelta(days=1)

    def _on_show_chart(self):
        if not self.ticks: return
        
        # Disable UI controls during chart processing to prevent hanging
        self._set_ui_enabled(False)
        
        try:
            tf = self.combo_tf.currentText()
            session = tickdata.SESSIONS[self.combo_session.currentText()]
            price = tickdata.PRICE_TYPES[self.combo_price.currentText()]
            tz_key = self.combo_tz.currentText()
            
            # 1. Completely reset finplot's internal state
            fplt.close() # This closes current internal windows
            
            # 2. Remove the old widget from our layout
            for i in reversed(range(self.chart_widget_layout.count())): 
                widget = self.chart_widget_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            
            # 3. Create a fresh plot
            self._apply_finplot_theme()
            self.ax_candle = fplt.create_plot(init_zoom_periods=100)
            self.chart_widget_layout.addWidget(self.ax_candle.vb.win)
            
            # 4. Process and plot data
            selected_session_key = tickdata.SESSIONS[self.combo_session.currentText()]
            
            df = None
            if tf == "Tick":
                # Do NOT filter by session for data building, only for highlighting
                df = self._build_tick_df(self.ticks, price, tz_key)
                if df is not None:
                    df.index = df.index.tz_localize(None)
                    self._draw_sessions(df, selected_session_key, tz_key)
                    fplt.plot(df["Price"], ax=self.ax_candle, color='#89b4fa')
                    self.lbl_chart_info.setText(f"Tick Data | {len(df)} points")
            else:
                target_tz_str = self.combo_tz.currentText()
                # Pass session=None to ticks_to_timeframe to avoid data filtering
                candles = tickdata.ticks_to_timeframe(self.ticks, tf, price, session=None, target_tz=target_tz_str)
                if candles:
                    # The candles might already be a list of dicts, let's ensure it's converted to DF correctly
                    df = pd.DataFrame(candles)
                    if not df.empty:
                        df['datetime'] = pd.to_datetime(df['datetime'], format="ISO8601")
                        df.set_index('datetime', inplace=True)
                        df.index = df.index.tz_localize(None)
                        
                        self._draw_sessions(df, selected_session_key, tz_key)
                        self.ax_volume = self.ax_candle.overlay()
                        fplt.candlestick_ochl(df[["open", "close", "high", "low"]], ax=self.ax_candle)
                        fplt.volume_ocv(df[["open", "close", "volume"]], ax=self.ax_volume)
                        self.lbl_chart_info.setText(f"{tf} Candles | {len(df)} bars")
            
            if df is not None and not df.empty:
                fplt.refresh()
                self.lbl_status_tf.setText(f"TF: {tf}")
                self.lbl_status_candles.setText(f"Candles: {len(df)}")
            else:
                self.lbl_chart_info.setText("No data to display for selected filters.")
                
        except Exception as e:
            print(f"Chart Error: {str(e)}")
            QMessageBox.critical(self, "Chart Error", f"Failed to draw chart: {str(e)}")
        finally:
            # Re-enable UI controls after processing
            self._set_ui_enabled(True)

    def _build_tick_df(self, ticks, price_type, tz_key):
        target_tz = tickdata.get_timezone_info(tz_key)
        converted_ticks = []
        for t in ticks:
            dt = t["datetime"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tickdata.ZoneInfo("UTC"))
            dt_converted = dt.astimezone(target_tz)
            price = round((t["ask"]+t["bid"])/2, 6) if price_type=="mid" else t.get(price_type, t["bid"])
            converted_ticks.append({
                "datetime": dt_converted,
                "Price": price
            })
        if not converted_ticks:
            return None
        df = pd.DataFrame(converted_ticks)
        df.set_index("datetime", inplace=True)
        return df

    @measure_time
    def _on_show_quantconnect(self):
        if not self.ticks:
            QMessageBox.warning(self, "QuantConnect", "No data loaded.")
            return
        self._set_ui_enabled(False)
        try:
            tf = self.combo_tf.currentText()
            price = tickdata.PRICE_TYPES[self.combo_price.currentText()]
            tz_key = self.combo_tz.currentText()
            target_tz_str = self.combo_tz.currentText()
            if tf == "Tick":
                tf_export = "5m"
            else:
                tf_export = tf
            candles = tickdata.ticks_to_timeframe(self.ticks, tf_export, price, session=None, target_tz=target_tz_str)
            if not candles:
                QMessageBox.warning(self, "QuantConnect", "No candle data for export.")
                return
            symbol = self.combo_symbol.currentText()
            first_dt = candles[0]["datetime"]
            last_dt = candles[-1]["datetime"]
            if isinstance(first_dt, str):
                first_dt = pd.to_datetime(first_dt, format="ISO8601")
            if isinstance(last_dt, str):
                last_dt = pd.to_datetime(last_dt, format="ISO8601")
            start_str = first_dt.strftime("%Y%m%d")
            end_str = last_dt.strftime("%Y%m%d")
            filename = f"{symbol}_{tf_export}_{start_str}_{end_str}_qc.parquet"
            filepath = os.path.join(self.last_save_dir, filename)

            candles_to_save = []
            for c in candles:
                item = {k: v for k, v in c.items() if k != "session"}
                candles_to_save.append(item)
            
            # Convert to DataFrame for Parquet saving
            df_candles = pd.DataFrame(candles_to_save)
            tickdata.save_candles(df_candles, filepath)
            fplt.close()
            for i in reversed(range(self.chart_widget_layout.count())):
                widget = self.chart_widget_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            self._apply_finplot_theme()
            self.ax_candle = fplt.create_plot(init_zoom_periods=100)
            self.chart_widget_layout.addWidget(self.ax_candle.vb.win)
            df = pd.DataFrame(candles)
            if not df.empty:
                df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
                df.set_index("datetime", inplace=True)
                df.index = df.index.tz_localize(None)
                selected_session_key = tickdata.SESSIONS[self.combo_session.currentText()]
                self._draw_sessions(df, selected_session_key, tz_key)
                self.ax_volume = self.ax_candle.overlay()
                fplt.candlestick_ochl(df[["open", "close", "high", "low"]], ax=self.ax_candle)
                fplt.volume_ocv(df[["open", "close", "volume"]], ax=self.ax_volume)
                self.lbl_chart_info.setText(f"QuantConnect {tf_export} | {len(df)} bars")
                fplt.refresh()
                QMessageBox.information(self, "QuantConnect", f"Exported {len(df)} candles to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "QuantConnect Error", str(e))
        finally:
            self._set_ui_enabled(True)

    def _on_save_csv(self):
        symbol = self.combo_symbol.currentText()
        default_name = f"{symbol}.parquet" if symbol else "data.parquet"
        initial_path = os.path.join(self.last_save_dir, default_name)
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Parquet", initial_path, "Parquet Files (*.parquet)")
        if filepath:
            # Convert ticks to DataFrame for Parquet saving
            df_ticks = pd.DataFrame(self.ticks)
            tickdata.save_ticks(df_ticks, filepath)
            QMessageBox.information(self, "Success", "Saved!")
            self.last_save_dir = os.path.dirname(filepath)

    @measure_time
    def _on_load_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Parquet", self.last_save_dir, "Parquet Files (*.parquet)")
        if filepath:
            self.ticks = tickdata.load_ticks_from_csv(filepath)  # Compatibility function that loads Parquet
            self.btn_show.setEnabled(True)
            self.btn_qc.setEnabled(True)

    def _on_open_download_manager(self):
        if self.data_center_window is None:
            self.data_center_window = TickDataDownloader(self.last_save_dir, self)
        self.data_center_window.show()
        self.data_center_window.raise_()
        self.data_center_window.activateWindow()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TickDataViewer()
    window.show()
    fplt.show(qt_exec=False)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
