# plotly_theme.py
import plotly.io as pio
import plotly.graph_objects as go


def register_minimal_theme():
    custom_font = "JetBrains Mono"
    base_template = pio.templates["plotly_white"].layout

    # Create new layout extending plotly_white
    custom_layout = go.Layout(
        font=dict(family=custom_font, size=12, color="black"),
        paper_bgcolor=base_template.paper_bgcolor,
        plot_bgcolor=base_template.plot_bgcolor,
        xaxis=base_template.xaxis,
        yaxis=base_template.yaxis,
        legend=base_template.legend,
        margin=base_template.margin,
    )

    # Register the new template
    pio.templates["minimal_custom"] = go.layout.Template(layout=custom_layout)


def set_minimal_theme_as_default():
    register_minimal_theme()
    pio.templates.default = "minimal_custom"
