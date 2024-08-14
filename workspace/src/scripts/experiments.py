import ipywidgets as widgets
from datetime import datetime
from IPython.display import display
from pathlib import Path
import shutil

class Experiments:
    def __init__(self):
        # Initialize the global variable to store the selected value
        self.selected_value = "pregenerated"

        # Load existing experiment directories, sorted alphabetically
        experiment_dirs = sorted(self.get_existing_experiments())

        # Create the dropdown widget with initial value and existing experiments
        self.dropdown = widgets.Dropdown(
            options=['pregenerated'] + experiment_dirs,
            value='pregenerated',
            description='Select experiment:',
        )

        # Observe changes in the dropdown selection
        self.dropdown.observe(self.on_change, names='value')

        # Create the button widget to add the current date and time
        self.new_button = widgets.Button(description='New Experiment')
        self.new_button.on_click(self.add_date)

        # Create the delete button
        self.delete_button = widgets.Button(description='Delete Experiment', button_style='danger')
        self.delete_button.on_click(self.confirm_delete_experiment)

        # Create an Output widget to display the confirmation dialog
        self.output = widgets.Output()

        # Display the dropdown, new button, delete button, and output
        display(self.dropdown, self.new_button, self.delete_button, self.output)

    def get_existing_experiments(self):
        """Return a list of existing experiment directories in '../outputs/generated/'."""
        generated_dir = Path("../outputs/generated/")

        if generated_dir.exists():
            # List all directories in the generated folder
            return [d.name for d in generated_dir.iterdir() if d.is_dir() and d.name != "check"]
        else:
            return []

    def on_change(self, change):
        """Update the selected value when the dropdown selection changes."""
        self.selected_value = change['new']

    def add_date(self, b):
        """Add the current date and time to the dropdown and set it as selected."""

        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.dropdown.options = list(self.dropdown.options) + [current_datetime]
        self.dropdown.value = current_datetime  # Set the new date-time as the selected value
        self.selected_value = current_datetime  # Update the global variable

        self.get_output_dir()

    def get_output_dir(self):
        """Create and return the appropriate output directory based on the selected value."""

        base_dir = Path("../outputs/")
        if self.selected_value == "pregenerated":
            output_dir = base_dir / "pregenerated"
        else:
            output_dir = base_dir / "generated" / self.selected_value

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def confirm_delete_experiment(self, b):
        """Prompt the user to confirm deletion of the selected experiment."""

        if self.selected_value == "pregenerated":
            with self.output:
                print("Cannot delete the 'pregenerated' experiment.")
            return

        with self.output:
            self.output.clear_output()
            print(f'Are you sure you want to delete the experiment "{self.selected_value}"?')
            confirm_button = widgets.Button(description='Confirm', button_style='danger')
            cancel_button = widgets.Button(description='Cancel')

            confirm_button.on_click(self.delete_experiment)
            cancel_button.on_click(lambda _: self.output.clear_output())

            display(widgets.HBox([confirm_button, cancel_button]))

    def delete_experiment(self, b):
        """Delete the selected experiment directory after confirmation."""

        dir_to_delete = Path("../outputs/generated/") / self.selected_value

        if dir_to_delete.exists() and dir_to_delete.is_dir():
            shutil.rmtree(dir_to_delete)
            with self.output:
                print(f'Experiment "{self.selected_value}" deleted.')

        # Update dropdown options
        new_options = sorted(self.get_existing_experiments())
        self.dropdown.options = ['pregenerated'] + new_options
        self.dropdown.value = 'pregenerated'
        self.selected_value = 'pregenerated'
        self.output.clear_output()

# Example of how to use the Experiments class
#experiment = Experiments()
