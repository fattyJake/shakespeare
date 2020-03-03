import os
import pickle
import traceback
from flask import Flask, jsonify, request, Response
from flask_swagger_ui import get_swaggerui_blueprint

from shakespeare import detect_api
from shakespeare import delete
from shakespeare import training

application = Flask(__name__)

### swagger specific ###
SWAGGER_URL = "/documentation"
API_URL = "/static/openapi.json"
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={"app_name": "RiskGapTargetingService"}
)
application.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###


# PATIENT_ID_LOG = 0
# DATA_CACHE = dict()


def check_content(content):
    if not content:
        return {
            "Request Status": "Failed",
            "Error Message": "Bad request: either no input or not "
            + "application/json type.",
        }
    else:
        return None


def check_payload(payload, correlation_id):
    if not payload:
        return {
            "Request Status": "Failed",
            "correlation_id": correlation_id,
            "request_api": "RiskGapTargetingService",
            "Error Message": "Bad request: no payload.",
        }
    else:
        return None


def check_model_version(model_version_ID, sub_type_id, correlation_id, update):
    if not isinstance(model_version_ID, int):
        return {
            "Request Status": "Failed",
            "correlation_id": correlation_id,
            "request_api": "RiskGapTargetingService",
            "Error Message": 'Bad request: plase provide "model_version_ID" as interger, '
            + f"got {model_version_ID} instead.",
        }
    else:
        if (
            not os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"shakespeare",
                    r"pickle_files",
                    r"ensembles",
                    f"ensemble_{model_version_ID}"
                    + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
                )
            )
            and update
        ):
            return None
        elif (
            os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"shakespeare",
                    r"pickle_files",
                    r"ensembles",
                    f"ensemble_{model_version_ID}"
                    + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
                )
            )
            and update
        ):
            return {
                "Request Status": "Failed",
                "correlation_id": correlation_id,
                "request_api": "RiskGapTargetingService",
                "Error Message": f'Bad request: model {model_version_ID}'
                + f"{'_' + str(sub_type_id) if sub_type_id else ''}"
                + " already exist, no need to update.",
            }
        elif (
            not os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"shakespeare",
                    r"pickle_files",
                    r"ensembles",
                    f"ensemble_{model_version_ID}"
                    + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
                )
            )
            and not update
        ):
            return {
                "Request Status": "Failed",
                "correlation_id": correlation_id,
                "request_api": "RiskGapTargetingService",
                "Error Message": f'Bad request: model {model_version_ID}'
                + f"{'_' + str(sub_type_id)} does "
                + "not exist, please provide correct number or update.",
            }
        else:
            return None


def check_mode(mode, correlation_id):
    if mode not in ['r', 'p', 'b']:
        return {
            "Request Status": "Failed",
            "correlation_id": correlation_id,
            "request_api": "RiskGapTargetingService",
            "Error Message": (
                '"mode" can only be one of "r", "p" or "b", '
                f'got {mode} instead.'
            ),
        }
    else:
        return None


@application.route("/", methods=["GET", "POST"])
def welcome():
    response_text = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
        <title>Risk Gap Targeting Service</title>
        <h2>Welcome to risk gap targeting service! Please view Swagger 
        documents under /documentation.</h2>
        <p></p>"""
    return response_text


@application.route("/detect", methods=["POST"])
def detect():
    content = request.get_json(silent=True)
    error_response = check_content(content)
    if error_response:
        return jsonify(error_response), 400

    correlation_id = content.get("correlation_id", "")

    try:
        payload = content.get("payload", {})
        error_response = check_payload(payload, correlation_id)
        if error_response:
            return jsonify(error_response), 400

        model_version_ID = content.get("model_version_ID", "null")
        sub_type_id = content.get("sub_type_id", None)
        error_response = check_model_version(
            model_version_ID, sub_type_id, correlation_id, False
        )
        if error_response:
            return jsonify(error_response), 400

        mode = content.get("mode", "b")
        error_response = check_mode(mode, correlation_id)
        if error_response:
            return jsonify(error_response), 400

        final_results = {
            "Request Status": "Success",
            "correlation_id": correlation_id,
            "target_year": content.get("target_year", "Not Provided"),
            "model_version_ID": model_version_ID,
            "sub_type_id": sub_type_id,
        }
        final_results.update(detect_api(content))

        return jsonify(final_results), 200
    except Exception as e:
        tb = traceback.format_exc().split("\n")
        error_response = {
            "Request Status": "Failed",
            "correlation_id": correlation_id,
            "Error": e.__class__.__name__,
            "Error Message": str(e),
            "stack_trace": "\n".join(tb),
        }
        return jsonify(error_response), 500


# @application.route("/update", methods=["PATCH"])
# def update_wrapper(model_version_ID):
#     content = request.get_json(silent=True)
#     error_response = check_content(content)
#     if error_response:
#         return jsonify(error_response), 400

#     correlation_id = content.get("correlation_id", "")

#     try:
#         payload = content.get("payload", {})
#         error_response = check_payload(payload, correlation_id)
#         if error_response:
#             return jsonify(error_response), 400

#         model_version_ID = content.get("model_version_ID", "null")
#         error_response = check_model_version(
#             model_version_ID, correlation_id, True
#         )
#         if error_response:
#             return jsonify(error_response), 400

#         print(
#             "############################## Training New Model "
#             + str(model_version_ID)
#             + " ##############################"
#         )
#         training_set = {
#             d["mem_id"]: list(
#                 set([c["code_type"] + "-" + c["code"] for c in d["codes"]])
#             )
#             for d in payload["training_set"]
#         }
#         mappings = {
#             d["code_type"] + "-" + d["code"]: d["hcc"]
#             for d in payload["mapping"]
#         }
#         pickle.dump(
#             mappings,
#             open(
#                 os.path.join(
#                     os.path.dirname(os.path.realpath(__file__)),
#                     r"pickle_files",
#                     r"mappings",
#                     f"mapping_{model_version_ID}",
#                 ),
#                 "wb",
#             ),
#         )
#         training.update_variables(training_set, model_version_ID)
#         training.update_ensembles(training_set, model_version_ID)
#         print(
#             "############################ Finished Training Model "
#             + str(model_version_ID)
#             + " ###########################"
#         )

#         return "Successful.", 200
#     except Exception as e:
#         tb = traceback.format_exc().split("\n")
#         error_response = {
#             "Request Status": "Failed",
#             "Error": e.__class__.__name__,
#             "Error Message": str(e),
#             "stack_trace": "\n".join(tb),
#         }
#         return jsonify(error_response), 500


# @application.route("/delete/<int:model_version_ID>", methods=["DELETE"])
# def delete_wrapper(model_version_ID):
#     try:
#         delete(model_version_ID)
#         return "Successful.", 200
#     except:
#         return "modelVersionID not found.", 404


# def periodcal_data_caching(data):
#     global DATA_CACHE, PATIENT_ID_LOG

#     DATA_CACHE[PATIENT_ID_LOG] = data
#     PATIENT_ID_LOG += 1

#     if len(DATA_CACHE) >= 1000:
#         now = str(datetime.now())
#         file_name = f"CARAMLService_API_{now.split('.')[0]}.json"
#         file_name = file_name.replace(' ', '_')
#         file_name = file_name.replace(':', '.')
#         file_name = os.path.join(
#             os.path.dirname(os.path.realpath(__file__)),
#             r"cache",
#             file_name,
#         )
#         json.dump(DATA_CACHE, open(file_name, 'w'))
#         DATA_CACHE = dict()
#         PATIENT_ID_LOG = 0


if __name__ == "__main__":
    application.run(host="0.0.0.0")
