from functools import partial
from ._ethogram import EthogramVizContainer, ETHOGRAM_COLORS
import pandas as pd
from fastplotlib import Plot
import numpy as np
from fastplotlib.graphics.selectors import LinearSelector, Synchronizer
from ipywidgets import HBox, VBox


class EthogramComparisonVizContainer(EthogramVizContainer):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_index: int = 0,
    ):
        """
        Creates container for editing ethograms and saving them to a new dataframe.

        Parameters
        ----------
        dataframe: ``pandas.Dataframe``
            Dataframe for ethograms that need to be cleaned. Should be organized in terms of `animal_id` and
            `session_id`. Should also have columns for comparing ethograms, those that were manually labeled and
            predictions.
        start_index: ``int``, default 0
            Row of the dataframe that will initially be selected to view videos and corresponding ethograms.
        """
        super(EthogramComparisonVizContainer, self).__init__(
            dataframe=dataframe, start_index=start_index, mode="inference"
        )

        if "ethograms" not in self._dataframe.columns:
            raise ValueError("In order to compare ground truth ethograms and predicted ethograms, "
                             "there must be hand-labeled ethograms in the dataframe.")

        self.comparison_plot = None

        self._make_ethogram_comparison_plot()

        self.synchronizer = Synchronizer(
            self.plot.selectors[0], self.comparison_plot.selectors[0], key_bind=None
        )

        self.plot.renderer.add_event_handler(
            partial(self._resize_plots, self.plot), "resize"
        )

    def _resize_plots(self, plot_instance, *args):
        """Event handler for making the ethogram plots resize together."""
        w, h = plot_instance.renderer.logical_size

        self.plot.canvas.set_logical_size(w, h)
        self.comparison_plot.canvas.set_logical_size(w, h)

    def _make_ethogram_comparison_plot(self):
        """Instantiates the ethogram plot."""
        row = self._dataframe.iloc[self.current_row_ix]

        if self.comparison_plot is None:
            self.comparison_plot = Plot(
                size=(500, 100), controller=self.plot.controller
            )

        self.ethogram_array = row["ethograms"]

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg = self.comparison_plot.add_line(
                data=np.column_stack([xs, ys]), thickness=20, name=b
            )

            lg.colors = 0
            lg.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -5) - 1
            lg.position_y = y_pos

        self.ethogram_selector = LinearSelector(
            selection=0,
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            parent=lg,
            end_points=(y_bottom, y_pos),
            arrow_keys_modifier=None,
        )

        self.comparison_plot.add_graphic(self.ethogram_selector)
        self.ethogram_selector.selection.add_event_handler(
            self._ethogram_selection_event_handler
        )
        self.comparison_plot.auto_scale()

    def _row_changed(self, *args):
        """Event handler for when a row in the datagrid is changed."""
        super()._row_changed()

        # force clearing of event handlers for selectors
        # seems to be an issue with fpl delete graphic method for selectors
        self.comparison_plot.selectors[0].selection._event_handlers.clear()
        self.comparison_plot.clear()
        self._make_ethogram_comparison_plot()

        del self.synchronizer
        self.synchronizer = Synchronizer(
            self.plot.selectors[0], self.comparison_plot.selectors[0], key_bind=None
        )

    def show(self):
        """Shows the widget."""
        trial_buttons = HBox([self.previous_button, self.next_button])

        return VBox(
            [
                self.datagrid,
                HBox([self.image_widget.show(), trial_buttons]),
                self.plot.show(),
                self.comparison_plot.show(),
            ]
        )
