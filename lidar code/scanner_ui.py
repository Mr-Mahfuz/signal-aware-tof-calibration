import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import serial
import serial.tools.list_ports
import threading
import time
import datetime
import os

class ScannerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LiDAR Scanner Control")
        self.root.geometry("600x450")

        self.serial_port = None
        self.is_connected = False
        self.is_scanning = False
        self.current_file = None
        self.scanned_points = 0

        self.setup_ui()
        self.refresh_ports()

    def setup_ui(self):
        # Top Frame - Connection
        conn_frame = ttk.LabelFrame(self.root, text="Connection", padding=10)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(conn_frame, text="COM Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(conn_frame, textvariable=self.port_var, width=15)
        self.port_cb.pack(side=tk.LEFT, padx=5)

        ttk.Button(conn_frame, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        # Middle Frame - Controls
        ctrl_frame = ttk.LabelFrame(self.root, text="Scan Control", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(ctrl_frame, text="Condition Label:").pack(side=tk.LEFT)
        self.label_var = tk.StringVar(value="unlabeled")
        ttk.Entry(ctrl_frame, textvariable=self.label_var, width=20).pack(side=tk.LEFT, padx=5)

        self.scan_btn = ttk.Button(ctrl_frame, text="Start Scan", command=self.start_scan, state=tk.DISABLED)
        self.scan_btn.pack(side=tk.LEFT, padx=5)

        # Bottom Frame - Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_txt = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_txt.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_txt.insert(tk.END, msg + "\n")
        self.log_txt.see(tk.END)

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_cb['values'] = ports
        if ports:
            self.port_cb.current(0)

    def toggle_connection(self):
        if not self.is_connected:
            port = self.port_var.get()
            if not port:
                messagebox.showerror("Error", "Select a COM port")
                return
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                self.is_connected = True
                self.connect_btn.config(text="Disconnect")
                self.scan_btn.config(state=tk.NORMAL)
                self.log(f"Connected to {port} at 115200 baud.")
                
                # Start reading thread
                self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
                self.read_thread.start()
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))
        else:
            self.is_connected = False
            if self.serial_port:
                self.serial_port.close()
            self.connect_btn.config(text="Connect")
            self.scan_btn.config(state=tk.DISABLED)
            self.log("Disconnected.")

    def start_scan(self):
        if not self.is_connected:
            return
        
        label = self.label_var.get().strip()
        if not label:
            label = "unlabeled"
            self.label_var.set(label)

        # Send the label to trigger the ESP32 scan
        self.serial_port.write((label + "\n").encode('utf-8'))
        self.scan_btn.config(state=tk.DISABLED)
        self.log(f"Sent command to start scan with label: {label}")

    def read_serial(self):
        while self.is_connected:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8', errors='replace').strip()
                    if not line:
                        continue

                    # Print minimal info to the UI log
                    if not self.is_scanning or not line[0].isdigit():
                        # Use self.root.after to safely update UI from thread
                        self.root.after(0, self.log, f"ESP32: {line}")
                    
                    if line == "SCAN_START":
                        self.is_scanning = True
                        self.scanned_points = 0
                        label = self.label_var.get()
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"scan_{timestamp}_{label}.csv"
                        
                        # Open a new file for this scan
                        self.current_file = open(filename, "w")
                        self.root.after(0, self.log, f"--> Started saving data to {filename}")

                    elif line == "SCAN_COMPLETE":
                        self.is_scanning = True
                        if self.current_file:
                            self.current_file.close()
                            self.current_file = None
                        self.root.after(0, self.log, f"--> Scan Complete! File saved.")
                        self.root.after(0, self.scan_btn.config, {"state": tk.NORMAL})
                        
                    else:
                        # If we are scanning and the line contains data (not a debug print), save it
                        if self.is_scanning and self.current_file:
                            self.current_file.write(line + "\n")
                            self.scanned_points += 1
                            if self.scanned_points % 20 == 0:
                                self.root.after(0, self.log, f"--> Progress: Scanned {self.scanned_points} points...")
                            
                            # Periodically flush so data isn't lost if unplugged
                            if self.scanned_points % 10 == 0:
                                self.current_file.flush()
                                os.fsync(self.current_file.fileno())
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerUI(root)
    root.mainloop()
