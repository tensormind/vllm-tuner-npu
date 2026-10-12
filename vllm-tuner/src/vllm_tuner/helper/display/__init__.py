import pandas as pd
from pydantic import BaseModel, field_validator
from rich.console import Console
from rich.table import Table


class DictDisplayModel(BaseModel):
    """Pydantic model to validate dictionary values for display."""

    data: dict[str, int | float | None | str]

    @field_validator("data")
    def validate_dict_values(cls, v):
        for key, value in v.items():
            if not isinstance(value, (int, float, type(None), str)):
                raise ValueError(f"Invalid type for key '{key}': {type(value).__name__}")
        return v


class DisplayConsole:
    """Console display class using Rich for formatted output."""

    def __init__(self):
        self.console = Console()

    def display_dict_as_table(
        self,
        data: dict[str, int | float | None | str],
        title: str | None = None,
        show_header: bool = True,
    ) -> None:
        """Display a dictionary as a formatted table.

        Args:
            data: Dictionary with string keys and int/float/None/str values
            title: Optional title for the table
            show_header: Whether to show column headers

        """
        try:
            # Validate data using Pydantic
            validated = DictDisplayModel(data=data)

            # Create table
            table = Table(title=title, show_header=show_header, show_lines=True)
            table.add_column("Key", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")

            # Add rows
            for key, value in validated.data.items():
                # Format the value
                if value is None:
                    value_str = "None"
                elif isinstance(value, float):
                    # Format floats with reasonable precision
                    value_str = f"{value:.6g}"
                else:
                    value_str = str(value)

                table.add_row(key, value_str)

            # Display the table
            self.console.print(table)

        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def display_df_as_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 20,
        max_col_width: int | None = 20,
        title: str | None = None,
        show_header: bool = True,
    ):
        """Display a pandas DataFrame using rich in a nicely formatted table.

        Args:
            df (pd.DataFrame): DataFrame to display
            max_rows (int): Maximum number of rows to display
            max_col_width (Optional[int]): Maximum width of each column. None for no limit
            title (str, optional): Optional title for the table
            show_header (bool): Whether to show column headers

        """
        try:
            table = Table(title=title, show_header=show_header, show_lines=True)

            # Add columns
            for col in df.columns:
                col_str = str(col)
                if max_col_width is not None and max_col_width > 0:
                    col_str = col_str[:max_col_width]
                table.add_column(col_str)

            # Limit rows
            rows_to_display = df.head(max_rows)

            for _, row in rows_to_display.iterrows():
                # Process each value in the row
                row_values = []
                for val in row:
                    val_str = str(val) if val is not None else ""
                    if max_col_width is not None and max_col_width > 0:
                        val_str = val_str[:max_col_width]
                    row_values.append(val_str)

                table.add_row(*row_values)

            self.console.print(table)
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def print(self, *args, **kwargs) -> None:
        self.console.print(*args, **kwargs)

    def display_trial_result(self, trial_data: dict[str, str], trial_number: int) -> None:
        """Display a trial result with a styled header."""
        table = Table(title=f"Trial #{trial_number}", show_header=True, show_lines=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        for key, value in trial_data.items():
            table.add_row(key, value)
        self.console.print(table)

    def display_study_summary(self, summary_data: dict[str, str], study_name: str = "Study") -> None:
        """Display a study summary with key statistics."""
        table = Table(title=f"Study Summary: {study_name}", show_header=True, show_lines=True)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        for key, value in summary_data.items():
            table.add_row(key, value)
        self.console.print(table)

    def display_progress_bar(self, current: int, total: int, label: str = "Progress") -> None:
        """Display a simple text-based progress indicator."""
        pct = (current / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        self.console.print(f"[bold]{label}:[/bold] [{bar}] {current}/{total} ({pct:.0f}%)")
