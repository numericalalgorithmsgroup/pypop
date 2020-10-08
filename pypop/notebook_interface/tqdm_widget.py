
from ipywidgets import HBox, HTML, Output, FloatProgress
from tqdm.notebook import tqdm_notebook

class tqdm_notebook_noauto(tqdm_notebook):

    @staticmethod
    def status_printer(_, total=None, desc=None, ncols=None):
        """
        Manage the printing of an IPython/Jupyter Notebook progress bar widget.
        """
        # Fallback to text bar if there's no total
        # DEPRECATED: replaced with an 'info' style bar
        # if not total:
        #    return super(tqdm_notebook, tqdm_notebook).status_printer(file)

        # fp = file

        if total:
            pbar = FloatProgress(min=0, max=total)
        else:  # No total? Show info style bar with no progress tqdm status
            pbar = FloatProgress(min=0, max=1)
            pbar.value = 1
            pbar.bar_style = 'info'
            if ncols is None:
                pbar.layout.width = "20px"

        ltext = HTML()
        rtext = HTML()
        if desc:
            ltext.value = desc
        container = HBox(children=[ltext, pbar, rtext])
        # Prepare layout
        if ncols is not None:  # use default style of ipywidgets
            # ncols could be 100, "100px", "100%"
            ncols = str(ncols)  # ipywidgets only accepts string
            try:
                if int(ncols) > 0:  # isnumeric and positive
                    ncols += 'px'
            except ValueError:
                pass
            pbar.layout.flex = '2'
            container.layout.width = ncols
            container.layout.display = 'inline-flex'
            container.layout.flex_flow = 'row wrap'

        return container 
