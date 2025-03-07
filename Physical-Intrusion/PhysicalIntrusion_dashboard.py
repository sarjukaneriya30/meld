"""Dashboard for Physical Intrusion using the Dash framework"""
import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional

import dash_bootstrap_components as dbc
import flask
import pandas as pd
from dash import Input, Output, dash_table, dcc, html
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Dash
from dotenv import load_dotenv
from flask import Flask, Response

from intrusiondet.bootstrapper import (
    IntrusionDetectionBootstrapper,
    bootstrap,
    create_sql_repository_connection,
)
from intrusiondet.core.colors import KPMG_BLUE
from intrusiondet.core.types import DashComponentID, LocationID
from intrusiondet.frontend.dashiohelpers import build_datatable_column
from intrusiondet.model.detectedobject import DetectedObject
from intrusiondet.model.savedvideo import SavedVideo

load_dotenv(os.fspath(Path(".env")))


HOME_PATH: Optional[str] = os.environ.get("INTRUSIONDET_PATH")
"""Path for configuration"""


if HOME_PATH is None:
    HOME_PATH = os.environ.get(
        "INTRUSIONDET_PATH", os.fspath(Path(__file__).parent.resolve())
    )

BOOTSTRAPPER: Final[IntrusionDetectionBootstrapper] = bootstrap(
    home_path=os.fspath(HOME_PATH),
    config_path=os.fspath(Path(HOME_PATH) / "configs/config.toml"),
)
"""This configures the program global settings"""


CAPTURES_DIR: Final[str] = Path(BOOTSTRAPPER.config.remote.video.captures_path).name
"""Directory of output videos from computer vision"""


CAMERA_LOCATION_ANY_STR: Final[str] = "All"


CAMERA_LOCATIONS: list[str] = [
    camera.location
    for camera in BOOTSTRAPPER.cameras.data.values()
]

CAMERA_LOCATIONS.insert(0, CAMERA_LOCATION_ANY_STR)


DASHAPP_GLOBAL_STYLES: dict[str, Any] = {"font-family": "Arial"}
"""Define some common styles"""


SERVER: Final[Flask] = Flask(__name__)
"""Flask server to manage images/videos"""


DASHAPP: Final[Dash] = Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO, dbc.icons.BOOTSTRAP],
    server=SERVER,
)
"""Dash application"""


KPMG_LOGO: Final[bytes] = base64.b64encode(
    Path("./assets/logos/kpmg-logo-png-transparent.png").open("rb").read()
)
"""KPMG logo, encoded"""


DATATABLE_COLUMNS_OLD_2_NEW: Final[dict[str, str]] = {
    "datetime_utc": "Date & Time (ZULU)",
    "class_name": "Object",
    "zones": "Zones",
    "filename": "filename",
}
"""Column names for the datatable of intrusions"""


class AppComponentIDs:
    """Store all the application component IDs"""

    reload_page: Final[DashComponentID] = "reload-page"
    """Selecting this component will reload the page"""

    pi_datatable: Final[DashComponentID] = "pi-datatable"
    """Physical Intrusion datatable"""

    pi_datatable_page_location: Final[DashComponentID] = "pi-datatable-location"
    """Physical Intrusion datatable location on the page"""

    pi_data_repository: Final[DashComponentID] = "raw-data-repository"
    """Dash store from data repository"""

    camera_location_dropdown: Final[DashComponentID] = "location-dropbox"
    """Dropdown option for restricting the datatable"""

    pi_video_player: Final[DashComponentID] = "pi-video-player"
    """Displays the video selected by the datatable"""


class AppLayoutComponents:
    """Dashboard components that are displayed"""

    nav_bar: Final[dbc.Navbar] = dbc.Navbar(
        [
            html.A(
                [
                    html.Img(
                        src="data:image/png;base64,{}".format(KPMG_LOGO.decode()),
                        style={"height": "2em", "padding-left": "1rem"},
                    ),
                ],
                id=AppComponentIDs.reload_page,
                href="/",
                style={"margin-right": "1rem"},
            ),
            html.Div(
                [
                    html.H1(
                        "Physical Intrusion Dashboard",
                        style={
                            "color": "white",
                            "margin-bottom": "0rem",
                            "width": "100%",
                            "font-family": "KPMG",
                        },
                    ),
                ],
                className="ml-auto",
                style={"padding-right": "1rem"},
            ),
        ],
        color=KPMG_BLUE.rbgstr(),
        sticky="top",
    )
    """Complete nav bar"""

    camera_location_dropdown: Final[html.Div] = html.Div(
        [
            html.Label("Select by Location"),
            dcc.Dropdown(
                CAMERA_LOCATIONS,
                value=CAMERA_LOCATION_ANY_STR,
                id=AppComponentIDs.camera_location_dropdown,
            ),
        ],
    )
    """Dropdown menu for the datatable"""

    pi_datatable: Final[html.Div] = html.Div(
        [
            dash_table.DataTable(
                id=AppComponentIDs.pi_datatable,
                page_size=10,
                row_selectable="single",
                style_as_list_view=True,  # removes vertical lines
                # style cell styles whole table
                style_cell={
                    "font-family": "Arial",
                    "font-size": "80%",
                    "textAlign": "left",
                    "height": "auto",
                    "whiteSpace": "normal",
                },
                style_header={"fontWeight": "bold"},
                style_data_conditional=[
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "inherit !important",
                        "border": "inherit !important",
                    },
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgba(0, 0, 0, 0.03)",
                    },
                ],
                css=[{"selector": ".row", "rule": "margin: 0"}],
            )
        ]
    )
    """Datatable highlighting providing entry for previewing videos for intrusion or
     not
    """

    video_playback: Final[html.Div] = html.Div(
        [
            html.Video(
                controls=True,
                muted=True,
                id=AppComponentIDs.pi_video_player,
                autoPlay=True,
                loop=True,
                className="vid-resize",
            )
        ]
    )
    """Video selected from datatable"""

    final_layout: Final[html.Div] = html.Div(
        [
            nav_bar,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            camera_location_dropdown,
                            html.Br(),
                            pi_datatable,
                        ],
                        width=6,
                    ),
                    dbc.Col([video_playback], style={"fontAlign": "center"}, width=6),
                ],
                justify="left",
            ),
            dcc.Location(id=AppComponentIDs.pi_datatable_page_location, refresh=False),
            dcc.Store(id=AppComponentIDs.pi_data_repository),
        ]
    )
    """Dash app layout"""


DASHAPP.layout = AppLayoutComponents.final_layout


@DASHAPP.callback(
    Output(AppComponentIDs.pi_data_repository, "data"),
    Input(AppComponentIDs.camera_location_dropdown, "value"),
    Input(AppComponentIDs.pi_datatable_page_location, "pathname"),
)
def get_raw_data_repository(
    camera_location_value: LocationID, page_update_location: str
) -> str:
    """From the value selected by the camera location dropbox, populate dash store"""
    if not page_update_location:
        raise PreventUpdate
    try:
        data_repo = create_sql_repository_connection(BOOTSTRAPPER.config)
        detections_query: list[DetectedObject] = data_repo.list(DetectedObject)
        detections: list[dict] = []

        det_obj_dict: dict[str, Any]
        for det_obj in detections_query:

            if camera_location_value != CAMERA_LOCATION_ANY_STR:
                if det_obj.location != camera_location_value:
                    continue

            # See if the same filename has been processed for the table
            detection_avail: dict = next(
                (
                    detect_dict
                    for detect_dict in detections
                    if detect_dict["filename"] == det_obj.filename
                ),
                None,
            )
            # If the filename was found, does this detection have an earlier datetime?
            if detection_avail is not None:
                if datetime.fromisoformat(
                    detection_avail[DATATABLE_COLUMNS_OLD_2_NEW["datetime_utc"]]
                ) > det_obj.datetime_utc.replace(tzinfo=timezone.utc):
                    detections.remove(detection_avail)
                else:
                    continue

            det_obj_dict = det_obj.asdict()

            # Remove any keys not defined above. return nothing
            _ = [
                det_obj_dict.pop(key)
                for key in det_obj_dict.copy().keys()
                if key not in DATATABLE_COLUMNS_OLD_2_NEW
            ]

            # Format as human friendly
            det_obj_dict[DATATABLE_COLUMNS_OLD_2_NEW["datetime_utc"]] = (
                datetime.fromtimestamp(det_obj_dict.pop("datetime_utc")).replace(
                    tzinfo=timezone.utc
                )
            ).isoformat()
            det_obj_dict[DATATABLE_COLUMNS_OLD_2_NEW["class_name"]] = det_obj_dict.pop(
                "class_name"
            )
            det_obj_dict[DATATABLE_COLUMNS_OLD_2_NEW["zones"]] = (
                det_obj_dict.pop("zones").strip("[").strip("]")
            )

            detections.append(det_obj_dict)

        detections.sort(
            key=lambda do: do[DATATABLE_COLUMNS_OLD_2_NEW["datetime_utc"]], reverse=True
        )
        dataframe_to_json: pd.DataFrame = pd.DataFrame(
            detections, columns=list(DATATABLE_COLUMNS_OLD_2_NEW.values())
        )
        return dataframe_to_json.to_json(date_format="iso", orient="split")
    except Exception as err:
        print("Unable to get data repository due to %s" % err)


@DASHAPP.callback(
    Output(AppComponentIDs.pi_datatable, "data"),
    Output(AppComponentIDs.pi_datatable, "columns"),
    Input(AppComponentIDs.pi_data_repository, "data"),
    Input(AppComponentIDs.pi_datatable_page_location, "pathname"),
)
def update_table(
    jsonified_data: str, location_update: Optional[str]
) -> tuple[list, list]:
    """From JSON string, get table and columns. Also check if location is updated"""
    if not location_update:
        raise PreventUpdate
    drop_columns: list[str] = ["filename", "Zones"]
    dataframe_from_json = pd.read_json(jsonified_data, orient="split").drop(
        columns=drop_columns, axis=1
    )

    out_columns = [
        build_datatable_column(col_name)
        for col_name in dataframe_from_json.columns
        if col_name not in drop_columns
    ]
    table_data = dataframe_from_json.to_dict("records")
    return table_data, out_columns


@DASHAPP.callback(
    Output(AppComponentIDs.pi_video_player, "src"),
    Input(AppComponentIDs.pi_data_repository, "data"),
    Input(AppComponentIDs.pi_datatable, "selected_rows"),
)
def play_video_after_user_selects_datatable_row(
    jsonified_data: str,
    selected_rows: list[int],
) -> Optional[str]:
    """After user selects row in datatable, get relevant video for playback"""
    if selected_rows is None:
        return None
    selected_row = selected_rows[0]
    dataframe_from_json = pd.read_json(jsonified_data, orient="split")
    try:
        series = dataframe_from_json.iloc[selected_row]
    except IndexError:
        return None
    filename = series[DATATABLE_COLUMNS_OLD_2_NEW["filename"]]
    sv = SavedVideo.reconstruct_from_filename(
        os.fspath(Path(BOOTSTRAPPER.config.remote.video.captures_path) / filename),
        BOOTSTRAPPER.config.video.schema,
        BOOTSTRAPPER.config.video.pattern,
        BOOTSTRAPPER.config.video.regex,
    )
    start_time = datetime.fromisoformat(
        series[DATATABLE_COLUMNS_OLD_2_NEW["datetime_utc"]]
    )
    start_time_sec = (start_time - sv.start_access_utc).total_seconds()
    ret_path = CAPTURES_DIR + "/%s#t=%f" % (
        series[DATATABLE_COLUMNS_OLD_2_NEW["filename"]],
        start_time_sec,
    )
    return ret_path


@SERVER.route("/" + CAPTURES_DIR + "/<path:path>")
def serve_static(path) -> Response:
    """Set path for videos to playback"""
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, CAPTURES_DIR), path)


def main() -> int:
    try:
        DASHAPP.run_server(
            host=os.getenv("DASH_HOST", "0.0.0.0"),
            port=os.getenv("DASH_PORT", "8050"),
            debug=bool(os.getenv("DASH_DEBUG", False)),
        )
    except (Exception,) as err:
        print(f"Dash app encountered {str(err)}")
    return 0


if __name__ == "__main__":
    main()
