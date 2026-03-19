import tkinter as tk
from tkinter import ttk
import subprocess
from PIL import Image, ImageTk
import os

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickSim")
        self.root.geometry("500x500+700+300")
        self.root.configure(bg='grey20')

        # Add title
        title_label = tk.Label(root, text="QuickSim", font=("Arial", 26, "bold"), bg='grey20', fg='white')
        title_label.pack(pady=20)

        # Add course and creators info
        info_label = tk.Label(root, text="Course: KETN01\nCreated by: Erina Binte Motahar and Zhuo Cheng", font=("Arial", 14), justify=tk.CENTER, bg='grey20', fg='white')
        info_label.pack(pady=10)

        # Add separator
        separator = tk.Frame(root, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)

        # Buttons
        self.button3 = tk.Button(root, text="Distillation Simulator", command=self.open_distillation, font=("Arial", 16), width=20, pady=10, padx=30, bg='white', fg='black', activebackground='lightgray')
        self.button3.pack(pady=15,padx=30)

        self.button2 = tk.Button(root, text="Chromatograph Simulator", command=self.open_chromatograph, font=("Arial", 16), width=20, pady=10, padx=30, bg='white', fg='black', activebackground='lightgray')
        self.button2.pack(pady=15,padx=30)

        self.button1 = tk.Button(root, text="Packed-bed Reactor Simulator", command=self.open_reactor, font=("Arial", 16), width=20, pady=10, padx=30, bg='white', fg='black', activebackground='lightgray')
        self.button1.pack(pady=15,padx=30) 




    def open_reactor(self):
        ReactorWindow()

    def open_chromatograph(self):
        ChromatographSimulator()

    def open_distillation(self):
        DistillationSimulator()

class ReactorWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Packed-bed Reactor Simulator")
        self.window.geometry("1200x800+400+150")
        self.window.configure(bg='grey20')

        # Left frame for inputs (non-scrollable)
        self.left_frame = tk.Frame(self.window, relief=tk.SUNKEN, bd=2, bg='grey20')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)

        # Store input entries
        self.input_entries = {}
        
        # Create input fields manually
        # Reactor Data section
        group_label = tk.Label(self.left_frame, text="Reactor Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)    
        self._create_input_field(self.left_frame, "Reactor Length", "L", "2", "m")
        self._create_input_field(self.left_frame, "Reactor Radius", "Rr", "0.5", "m")
        self._create_input_field(self.left_frame, "Void", "eps_c", "0.33", "")
        self._create_input_field(self.left_frame, "Porosity", "poro", "0.5", "")
        # Add separator
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)        
       
        # Feed Data section
        group_label = tk.Label(self.left_frame, text="Feed Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "Feed Conc", "cAin", "25", "mol/m3")
        self._create_input_field(self.left_frame, "Feed temperature", "Tin", "353", "k")
        self._create_input_field(self.left_frame, "Feed flow rate", "fflow", "50", "m3/h")
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)  
        
        # Reaction Data section
        group_label = tk.Label(self.left_frame, text="Reaction Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "Frequency Factor", "A", "240000000", "/s")
        self._create_input_field(self.left_frame, "Activation Energy", "E", "70000", "J/mol")
        self._create_input_field(self.left_frame, "Reaction Heat", "deltaHr", "-180000", "J/mol")
        self._create_input_field(self.left_frame, "Gas density", "rho", "4", "kg/m3")
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)  
        
        # Add plot selection checkboxes
        plot_label = tk.Label(self.left_frame, text="Select Output Plots", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        plot_label.pack(pady=(15, 8), padx=5, anchor=tk.W)
        
        # Create a frame for the checkbox grid
        checkbox_frame = tk.Frame(self.left_frame, bg='grey20')
        checkbox_frame.pack(pady=10, padx=10)
        
        # Create variables for plot checkboxes
        self.plot_vars = {}
        plot_names = ["Conc & Temp 2D", "Concentration 3D", "Temperature 3D", "Flow Profile", "Mean Conc & Temp", "Dynamic plot"]
        for i, plot_name in enumerate(plot_names, 1):
            print(i)
            var = tk.BooleanVar(value=True)
            self.plot_vars[f"fig{i}"] = var
            # Calculate row and column for 3x2 grid (3 columns, 2 rows)
            row = (i - 1) // 3
            col = (i - 1) % 3
            checkbox = tk.Checkbutton(checkbox_frame, text=plot_name, variable=var, bg='grey20', fg='white', selectcolor='grey20')
            checkbox.grid(row=row, column=col, pady=5, padx=5, sticky=tk.W)
            print(row, col)
        
        # Add simulate button at the bottom of inputs
        self.calc_button = tk.Button(self.left_frame, text="Simulate and Plot", command=self.calculate_and_plot, font=("Arial", 14), padx=15, pady=10, bg='white', fg='grey20', activebackground='lightgray')
        self.calc_button.pack(pady=20, anchor=tk.CENTER)

        # Right frame for scrollable output
        self.right_frame = tk.Frame(self.window, bg='grey20')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create scrollable canvas for output
        self.canvas = tk.Canvas(self.right_frame, bg='grey20', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='grey20')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # Store referenced images to prevent garbage collection
        self.images_ref = []

    def _create_input_field(self, parent, label_text, var_name, default_value, unit):
        """Helper method to create an input field with label, entry, and unit"""
        param_frame = tk.Frame(parent, bg='grey20')
        param_frame.pack(fill=tk.X, pady=5, padx=10)
        
        # Parameter name label
        tk.Label(param_frame, text=label_text, width=25, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
        
        # Entry field with default value
        entry = tk.Entry(param_frame, width=12, bg='white', fg='grey20')
        entry.insert(0, default_value)
        entry.pack(side=tk.LEFT, padx=5)
        
        # Unit label
        tk.Label(param_frame, text=unit, width=15, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
        
        # Store the entry widget for later retrieval
        self.input_entries[var_name] = entry

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    def clear_output(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.images_ref = []

    def add_text(self, text):
        label = tk.Label(self.scrollable_frame, text=text, font=("Arial", 14), wraplength=400, justify=tk.LEFT, bg='grey20', fg='white')
        label.pack(pady=5, padx=5, fill=tk.X)

    def add_image(self, image_path, width=400, height=250):
        try:
            img = Image.open(image_path)
            # Resize the image to exact dimensions (will enlarge if needed)
            img = img.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.images_ref.append(photo)  # Keep reference
            image_label = tk.Label(self.scrollable_frame, image=photo, bg='grey20')
            image_label.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)
        except Exception as e:
            error_label = tk.Label(self.scrollable_frame, text=f"Error loading {image_path}: {str(e)}", fg="red", bg='grey20')
            error_label.pack(pady=5, padx=5)

    def calculate_and_plot(self):
        try:
            # Collect all parameter values
            params = {}
            for var_name, entry in self.input_entries.items():
                params[var_name] = float(entry.get())
            
            # Clear previous output
            self.clear_output()
            #self.add_text("Running simulation...\n")
            
            # Create command to run reactor_backend.py with parameters
            cmd = ['python', 'backend/reactor_backend.py']
            for var_name, value in params.items():
                cmd.extend([f'--{var_name}', str(value)])
            
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode == 0:
                # Display output from reactor_backend
                if result.stdout:
                    self.add_text("\n" + result.stdout)
                
                # Display only selected plots based on checkboxes
                plot_files = [
                    ('backend/plots/reactor_fig1.png', 'fig1'),
                    ('backend/plots/reactor_fig2.png', 'fig2'),
                    ('backend/plots/reactor_fig3.png', 'fig3'),
                    ('backend/plots/reactor_fig4.png', 'fig4'),
                    ('backend/plots/reactor_fig5.png', 'fig5'),
                    ('backend/plots/reactor_fig6.png', 'fig6')
                ]
                for plot_file, fig_key in plot_files:
                    # Only display if the checkbox is selected
                    if self.plot_vars[fig_key].get():
                        plot_path = os.path.join(os.path.dirname(__file__), plot_file)
                        if os.path.exists(plot_path):
                            #self.add_text(f"\n{plot_file}:")
                            self.add_image(plot_path, width=450, height=300)
            else:
                self.add_text("Error running simulation:\n" + result.stderr)
        except ValueError as e:
            self.clear_output()
            self.add_text(f"Invalid input. Please enter valid numbers.\nError: {str(e)}")


#-------------------------------------------------------------------------------------------------


class ChromatographSimulator:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Chromatograph Simulator")
        self.window.geometry("1150x750+350+150")
        self.window.configure(bg='grey20')

        # Left frame for inputs (non-scrollable)
        self.left_frame = tk.Frame(self.window, relief=tk.SUNKEN, bd=2, bg='grey20')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)

        # Store input entries
        self.input_entries = {}
        
        # Column Data section
        group_label = tk.Label(self.left_frame, text="Column Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)    
        self._create_input_field(self.left_frame, "Column Length", "L", "10.0", "cm")
        self._create_input_field(self.left_frame, "Column Void Fraction", "eps_c", "0.45", "")
        self._create_input_field(self.left_frame, "Packing Porosity", "eps_p", "0.57", "")
        self._create_input_field(self.left_frame, "Particle Diameter", "dp", "0.003", "cm")
        
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Flow Parameters section
        group_label = tk.Label(self.left_frame, text="Flow Parameters", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "Peclet Number", "Pe", "0.5", "")
        self._create_input_field(self.left_frame, "Superficial Velocity", "v", "300", "cm/min")
        self._create_input_field(self.left_frame, "CV/h", "F", "30", "")
        
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Simulation Parameters section
        group_label = tk.Label(self.left_frame, text="Simulation Parameters", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "Grid Points", "N", "120", "")
        self._create_input_field(self.left_frame, "Loading Time", "t_load", "0.5", "CV")
        self._create_input_field(self.left_frame, "Max Time", "t_max", "56.0", "CV")
        
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Feed Concentration section
        group_label = tk.Label(self.left_frame, text="Feed Concentrations", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        
        # Create a frame for the 2x2 grid
        feed_conc_frame = tk.Frame(self.left_frame, bg='grey20')
        feed_conc_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # Define components and their parameters
        components = [
            ("Component A", "c_feed_A", "1.15"),
            ("Component B", "c_feed_B", "2.0"),
            ("Component C", "c_feed_C", "0.23"),
            ("Component D", "c_feed_D", "0.40")
        ]
        
        # Create 2x2 grid
        for i, (label_text, var_name, default_value) in enumerate(components):
            row = i // 2
            col = i % 2
            
            # Create frame for this component
            param_frame = tk.Frame(feed_conc_frame, bg='grey20')
            param_frame.grid(row=row, column=col, pady=5, padx=5, sticky=tk.W)
            
            # Parameter name label
            tk.Label(param_frame, text=label_text, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
            
            # Entry field
            entry = tk.Entry(param_frame, width=10, bg='white', fg='grey20')
            entry.insert(0, default_value)
            entry.pack(side=tk.LEFT, padx=3)
            
            # Unit label
            tk.Label(param_frame, text="g/L", anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT, padx=3)
            
            # Store entry
            self.input_entries[var_name] = entry
        
        # Add simulate button at the bottom of inputs
        self.calc_button = tk.Button(self.left_frame, text="Simulate and Plot", command=self.calculate_and_plot, font=("Arial", 14), padx=15, pady=10, bg='white', fg='grey20', activebackground='lightgray')
        self.calc_button.pack(pady=20, anchor=tk.CENTER)

        # Right frame
        self.right_frame = tk.Frame(self.window, bg='grey20')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top part - Output frame
        self.output_frame = tk.Frame(self.right_frame, relief=tk.SUNKEN, bd=2, bg='grey20')
        self.output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create static frame for output
        self.scrollable_frame = tk.Frame(self.output_frame, bg='grey20')
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)

        # Bottom part - Input frame for future use
        self.bottom_input_frame = tk.Frame(self.right_frame, relief=tk.SUNKEN, bd=2, bg='grey20')
        self.bottom_input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)

        # Add label for component parameters
        bottom_label = tk.Label(self.bottom_input_frame, text="Component Parameters", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        bottom_label.pack(pady=(8, 5), padx=5, anchor=tk.W)

        # Create table for component parameters
        self._create_component_parameters_table(self.bottom_input_frame)

        # Store referenced images to prevent garbage collection
        self.images_ref = []

    def _create_component_parameters_table(self, parent):
        """Create a table for component parameters (H0, beta, q_max, k_kin)"""
        # Initialize dictionary to store component parameter entries
        self.component_params = {}
        
        # Default values for each component
        defaults = {
            'H0': [0.6e-4, 8.5e-4, 2.97e-4, 9e-4],
            'beta': [4.37, 5.17, 5.14, 5.93],
            'q_max': [70.0, 70.0, 70.0, 70.0],
            'k_kin': [300.0, 75.0, 90.0, 42.0]
        }
        
        components = ['A', 'B', 'C', 'D']
        param_names = ['H0', 'beta', 'q_max', 'k_kin']
        
        # Create table frame with grid
        table_frame = tk.Frame(parent, bg='grey20')
        table_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # Column headers
        tk.Label(table_frame, text="Component", width=12, anchor=tk.W, font=("Arial", 10, "bold"), bg='grey20', fg='white').grid(row=0, column=0, padx=3, pady=3)
        for col, param_name in enumerate(param_names, start=1):
            tk.Label(table_frame, text=param_name, width=12, anchor=tk.CENTER, font=("Arial", 10, "bold"), bg='grey20', fg='white').grid(row=0, column=col, padx=3, pady=3)
        
        # Row data
        for row, comp in enumerate(components, start=1):
            tk.Label(table_frame, text=comp, width=12, anchor=tk.W, font=("Arial", 10, "bold"), bg='grey20', fg='white').grid(row=row, column=0, padx=3, pady=3)
            
            for col, param_name in enumerate(param_names, start=1):
                entry = tk.Entry(table_frame, width=10, bg='white', fg='grey20')
                entry.insert(0, str(defaults[param_name][row-1]))
                entry.grid(row=row, column=col, padx=3, pady=3)
                
                # Store entry in dictionary with key like 'H0_A', 'H0_B', etc.
                self.component_params[f"{param_name}_{comp}"] = entry

    def _create_input_field(self, parent, label_text, var_name, default_value, unit):
        """Helper method to create an input field with label, entry, and unit"""
        param_frame = tk.Frame(parent, bg='grey20')
        param_frame.pack(fill=tk.X, pady=5, padx=10)
        
        # Parameter name label
        tk.Label(param_frame, text=label_text, width=25, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
        
        # Entry field with default value
        entry = tk.Entry(param_frame, width=12, bg='white', fg='grey20')
        entry.insert(0, default_value)
        entry.pack(side=tk.LEFT, padx=5)
        
        # Unit label
        tk.Label(param_frame, text=unit, width=15, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
        
        # Store the entry widget for later retrieval
        self.input_entries[var_name] = entry

    def clear_output(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.images_ref = []

    def add_text(self, text):
        label = tk.Label(self.scrollable_frame, text=text, font=("Arial", 14), wraplength=600, justify=tk.LEFT, bg='grey20', fg='white')
        label.pack(pady=2, padx=5, fill=tk.X)

    def add_image(self, image_path, width=600, height=350):
        try:
            img = Image.open(image_path)
            # Resize the image to exact dimensions (will enlarge if needed)
            img = img.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.images_ref.append(photo)  # Keep reference
            image_label = tk.Label(self.scrollable_frame, image=photo, bg='grey20')
            image_label.pack(pady=2, padx=5, fill=tk.BOTH, expand=True)
        except Exception as e:
            error_label = tk.Label(self.scrollable_frame, text=f"Error loading {image_path}: {str(e)}", fg="red", bg='grey20')
            error_label.pack(pady=5, padx=5)

    def calculate_and_plot(self):
        try:
            # Collect all parameter values
            params = {}
            int_params = {'N'}  # Parameters that should be integers
            for var_name, entry in self.input_entries.items():
                value = entry.get()
                if var_name in int_params:
                    params[var_name] = int(float(value))  # Convert to float first, then int
                else:
                    params[var_name] = float(value)
            
            # Collect component parameters
            for param_key, entry in self.component_params.items():
                params[param_key] = float(entry.get())
            
            # Clear previous output
            self.clear_output()
            #self.add_text("Simulation Result")
            
            # Create command to run chromatography_backend.py with parameters
            cmd = ['python', 'backend/chromatography_backend.py']
            for var_name, value in params.items():
                cmd.extend([f'--{var_name}', str(value)])
            
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode == 0:
                # Display output from chromatography_backend
                if result.stdout:
                    self.add_text(result.stdout)
                
                # Display the plot
                plot_file = 'backend/plots/chrom_fig1.png'
                plot_path = os.path.join(os.path.dirname(__file__), plot_file)
                if os.path.exists(plot_path):
                    self.add_image(plot_path, width=550, height=350)
            else:
                self.add_text("Error running simulation:\n" + result.stderr)
                print(result.stderr)
        except ValueError as e:
            self.clear_output()
            self.add_text(f"Invalid input. Please enter valid numbers.\nError: {str(e)}")

class DistillationSimulator:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Distillation Simulator")
        self.window.geometry("1000x650+450+250")
        self.window.configure(bg='grey20')

        # Left frame for inputs (non-scrollable)
        self.left_frame = tk.Frame(self.window, relief=tk.SUNKEN, bd=2, bg='grey20')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)

        # Store input entries
        self.input_entries = {}
        
        # Feed Data section
        group_label = tk.Label(self.left_frame, text="Feed Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "Feed Flow Rate", "F", "2268", "kmole/h")
        self._create_input_field(self.left_frame, "Feed Mole Fraction", "xfeed", "0.144", "")
        
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Column Data section
        group_label = tk.Label(self.left_frame, text="Column Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "Number of Trays", "N", "20", "")
        self._create_input_field(self.left_frame, "Feed Tray Location", "Nf", "18", "")
        self._create_input_field(self.left_frame, "Tray Hold-up", "M", "20", "kmole")
        self._create_input_field(self.left_frame, "Condenser Hold-up", "Mc", "500", "kmole")
        self._create_input_field(self.left_frame, "Boiler Hold-up", "Mb", "200", "kmole")
        
        separator = tk.Frame(self.left_frame, height=2, bd=1, relief=tk.SUNKEN, bg='gray')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Operation Data section
        group_label = tk.Label(self.left_frame, text="Operation Data", font=("Arial", 14, "bold"), bg='grey20', fg='white')
        group_label.pack(pady=(8, 2), padx=5, anchor=tk.W)
        self._create_input_field(self.left_frame, "System Pressure", "P", "101300", "Pa")
        self._create_input_field(self.left_frame, "Reflux Ratio", "RD", "1.66", "")
        self._create_input_field(self.left_frame, "Vapor Flow", "V", "1027", "kmole/h")
        self._create_input_field(self.left_frame, "Simulation Time", "t_max", "200", "hours")
        
        # Add simulate button at the bottom of inputs
        self.calc_button = tk.Button(self.left_frame, text="Simulate and Plot", command=self.calculate_and_plot, font=("Arial", 14), padx=15, pady=10, bg='white', fg='grey20', activebackground='lightgray')
        self.calc_button.pack(pady=20, anchor=tk.CENTER)

        # Right frame for scrollable output
        self.right_frame = tk.Frame(self.window, bg='grey20')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create scrollable canvas for output
        self.canvas = tk.Canvas(self.right_frame, bg='grey20', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='grey20')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # # Enable mouse wheel scrolling
        # self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        # self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # Store referenced images to prevent garbage collection
        self.images_ref = []

    def _create_input_field(self, parent, label_text, var_name, default_value, unit):
        """Helper method to create an input field with label, entry, and unit"""
        param_frame = tk.Frame(parent, bg='grey20')
        param_frame.pack(fill=tk.X, pady=5, padx=10)
        
        # Parameter name label
        tk.Label(param_frame, text=label_text, width=25, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
        
        # Entry field with default value
        entry = tk.Entry(param_frame, width=12, bg='white', fg='grey20')
        entry.insert(0, default_value)
        entry.pack(side=tk.LEFT, padx=5)
        
        # Unit label
        tk.Label(param_frame, text=unit, width=15, anchor=tk.W, bg='grey20', fg='white').pack(side=tk.LEFT)
        
        # Store the entry widget for later retrieval
        self.input_entries[var_name] = entry

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    def clear_output(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.images_ref = []

    def add_text(self, text):
        label = tk.Label(self.scrollable_frame, text=text, font=("Arial", 11), wraplength=400, justify=tk.LEFT, bg='grey20', fg='white')
        label.pack(pady=5, padx=5, fill=tk.X)

    def add_image(self, image_path, width=400, height=300):
        try:
            img = Image.open(image_path)
            # Resize the image to exact dimensions (will enlarge if needed)
            img = img.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.images_ref.append(photo)  # Keep reference
            image_label = tk.Label(self.scrollable_frame, image=photo, bg='grey20')
            image_label.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)
        except Exception as e:
            error_label = tk.Label(self.scrollable_frame, text=f"Error loading {image_path}: {str(e)}", fg="red", bg='grey20')
            error_label.pack(pady=5, padx=5)

    def calculate_and_plot(self):
        try:
            # Collect all parameter values
            params = {}
            int_params = {'N', 'Nf', 'P'}  # Parameters that should be integers
            for var_name, entry in self.input_entries.items():
                value = entry.get()
                if var_name in int_params:
                    params[var_name] = int(float(value))  # Convert to float first, then int
                else:
                    params[var_name] = float(value)
            
            # Clear previous output
            self.clear_output()
            #self.add_text("Simulation Complete\n")
            
            # Create command to run dist_backend_2.py with parameters
            cmd = ['python', 'backend/dist_backend_2.py']
            for var_name, value in params.items():
                cmd.extend([f'--{var_name}', str(value)])
            
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode == 0:
                # Display output from dist_backend_2
                if result.stdout:
                    self.add_text(result.stdout)
                
                # Display the plot
                plot_file = 'backend/plots/dist_fig1.png'
                plot_path = os.path.join(os.path.dirname(__file__), plot_file)
                if os.path.exists(plot_path):
                    self.add_image(plot_path, width=300, height=600)
            else:
                self.add_text("Error running simulation:\n" + result.stderr)
        except ValueError as e:
            self.clear_output()
            self.add_text(f"Invalid input. Please enter valid numbers.\nError: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()