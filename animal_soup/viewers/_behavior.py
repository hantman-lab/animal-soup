import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import HBox, VBox, Button, Layout
from fastplotlib import ImageWidget
from warnings import warn
from ..arrays import LazyVideo
from typing import *
from ..utils import get_parent_raw_data_path, resolve_path
from decord import gpu as gpu_context

DECORD_CONTEXT = "cpu"


class BehaviorVizContainer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
    ):
        """
        Creates an ipydatagrid and `fastplotlib` ``ImageWidget`` viewer based on a dataframe.

        Parameters
        ----------
        dataframe: pd.DataFrame
            Dataframe used to create the ipydatagrid. Datagrid will show a subset of columns from the original
            dataframe.
        start_index: int, default 0
            Row index in datagrid that will start out as being selected initially. Default is first row.
        """
        self._dataframe = dataframe
        self.local_parent_path = get_parent_raw_data_path()

        hide_columns = [
            "ethograms",
            "cleaned_ethograms",
            "notes",
            "exp_type",
            "model_params",
            "vid_paths",
            "output_path"
        ]

        columns = dataframe.columns

        default_widths = {"animal_id": 200, "session_id": 100, "trial_id": 200}

        df_show = self._dataframe[[c for c in columns if c not in hide_columns]]

        self.datagrid = DataGrid(
            df_show,
            selection_mode="cell",
            layout={"height": "250px", "width": "750px"},
            base_row_size=24,
            index_name="index",
            column_widths=default_widths,
        )

        self.current_row_ix: int = start_index
        self.trial_selector = None
        self.image_widget = None

        self.datagrid.select(
            row1=start_index, column1=0, row2=start_index, column2=0, clear_mode="all"
        )
        self._make_image_widget()

        self.previous_button = Button(description="Previous",
                                      disabled=False,
                                      value=False,
                                      layout=Layout(width="auto"),
                                      tooltip='previous trial',
                                      icon='long-arrow-alt-left')
        self.next_button = Button(description="Next",
                                  layout=Layout(width="auto"),
                                  disabled=False,
                                  value=False,
                                  tooltip='next trial',
                                  icon='long-arrow-alt-right')

        self.datagrid.observe(self._row_changed, names="selections")
        self.previous_button.on_click(self._previous_trial)
        self.next_button.on_click(self._next_trial)

    def _get_selection_row(self) -> Union[int, None]:
        """Returns the index of the current selected row."""
        r1 = self.datagrid.selections[0]["r1"]
        r2 = self.datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        index = self.datagrid.get_visible_data().index[r1]

        return index

    def _row_changed(self, *args):
        """Event handler for when a row in the datagrid is changed."""
        index = self._get_selection_row()

        if index is None or self.current_row_ix == index:
            return

        self.current_row_ix = index

        if self.image_widget is None:
            self._make_image_widget()
        else:
            self._update_image_widget()

    def _make_image_widget(self):
        """Instantiates image widget to view behavior videos."""
        row = self._dataframe.iloc[self.current_row_ix]

        front_vid_path = row["vid_paths"]["front"]
        side_vid_path = row["vid_paths"]["side"]

        if DECORD_CONTEXT == "gpu":
            data = [LazyVideo(resolve_path(side_vid_path), ctx=gpu_context(0)),
                    LazyVideo(resolve_path(front_vid_path), ctx=gpu_context(0))]
        else:
            data = [LazyVideo(resolve_path(side_vid_path)),
                    LazyVideo(resolve_path(front_vid_path))]

        if self.image_widget is None:
            self.image_widget = ImageWidget(data=data,
                                            grid_plot_kwargs={"size": (700, 300)})

    def _update_image_widget(self):
        """If row changes, update the data in the ImageWidget with the new row selected."""
        row = self._dataframe.iloc[self.current_row_ix]

        front_vid_path = row["vid_paths"]["front"]
        side_vid_path = row["vid_paths"]["side"]

        if DECORD_CONTEXT == "gpu":
            data = [LazyVideo(resolve_path(side_vid_path), ctx=gpu_context(0)),
                    LazyVideo(resolve_path(front_vid_path), ctx=gpu_context(0))]
        else:
            data = [LazyVideo(resolve_path(side_vid_path)),
                    LazyVideo(resolve_path(front_vid_path))]

        self.image_widget.set_data(data,
                                   reset_vmin_vmax=True)

    def _previous_trial(self, obj):
        """Event handler to go to the previous trial on click."""
        self.datagrid.selections.clear()
        if self.current_row_ix - 1 < 0:
            self.datagrid.selections = [{'r1': len(self._dataframe.index) - 1, 'c1': 0, 'r2': len(self._dataframe.index) - 1, 'c2': 0}]
        else:
            self.datagrid.selections = [
                {'r1': self.current_row_ix - 1, 'c1': 0, 'r2': self.current_row_ix - 1, 'c2': 0}]

        self._row_changed()

    def _next_trial(self, obj):
        """Event handler to go to the next trial on click."""
        self.datagrid.selections.clear()
        if self.current_row_ix + 1 > len(self._dataframe.index) - 1:
            self.datagrid.selections = [
                {'r1': 0, 'c1': 0, 'r2': 0, 'c2': 0}]
        else:
            self.datagrid.selections = [
                {'r1': self.current_row_ix + 1, 'c1': 0, 'r2': self.current_row_ix + 1, 'c2': 0}]

        self._row_changed()

    def show(self):
        """Shows the widget."""
        # box trial buttons together
        trial_buttons = HBox([self.previous_button, self.next_button])

        # return widget of all elements together
        return VBox(
            [self.datagrid, HBox([self.image_widget.show(sidecar=False), trial_buttons])]
        )
