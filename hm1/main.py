from flask import Flask
from flask_restx import Api, Resource, reqparse, abort
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
import ast
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="My REST API",
    description="REST API allow to do operations on the ML model",
    default="Operations on an ML model",
    default_label="available: train, predict, delete"

)

req_post_args = reqparse.RequestParser()
req_post_args.add_argument("task", type=str, location='args',
                           help='type of task: classification/regression',
                           required=True)
req_post_args.add_argument("model", type=str, location='args',
                           help='model is required for training',
                           required=True)
req_post_args.add_argument("path_model", type=str, location='args',
                           help='path of model is required',
                           required=True)
req_post_args.add_argument(
    "params", type=str, location='args', help='params of model',
    required=False)
req_post_args.add_argument("file", location='files',
                           type=FileStorage, required=True)

req_post_args_file = reqparse.RequestParser()
req_post_args_file.add_argument(
    "file",
    location='files',
    type=FileStorage,
    required=True
)
req_post_args_file.add_argument(
    "path_model",
    type=str,
    location='args',
    help='path of model is required',
    required=True
)

req_delete = reqparse.RequestParser()
req_delete.add_argument("path_model", type=str, location='args',
                           help='path of model is required',
                           required=True)

def error(type_for_check, check_param, task=None):
    if type_for_check == 'task':
        if check_param not in ['classification', 'regression']:
            abort(message='incorrect task')
    if type_for_check == 'model':
        if task == 'classification':
            if check_param not in ['logreg', 'xgb_classifier']:
                abort(message='incorrect name model')
        else:
            if check_param not in ['linreg', 'xgb_regressor']:
                abort(message='incorrect name model')
    if type_for_check == 'path_model':
        if check_param not in os.listdir('models'):
            abort(message='incorrect path of model')


@api.route('/model', methods=['POST', 'DELETE'])
class TrainModel(Resource):
    @api.expect(req_post_args)
    @api.doc(params={
        'task': 'Choose type of your task - classification or regression',
        'model': 'Classification: LogisticRegression (logreg) or XGBoostClassifier \
        (xgb_classifier). Regression: LinearRegression (linreg) or \
         XGBRegressor (xgb_regressor)',
        'params': "Params for training model",
        'path_model': 'Unique name of your model',
        'file': "Upload file for training model. \
        File must contain a column name as 'target'"
    }
    )
    @api.response(201, description='model successfully created')
    def post(self):
        """
        Train model
        """
        request_data = req_post_args.parse_args()
        # get input param
        task = request_data.get('task')
        path_model = request_data.get('path_model')
        dict_parameters = request_data.get('parameters')

        if dict_parameters is None:
            dict_parameters = dict()
        else:
            dict_parameters = ast.literal_eval(dict_parameters)
        model = request_data.get('model')
        error('task', task)
        error('model', model, task)

        data_train = pd.read_csv(request_data['file'])
        y, x = data_train['target'], data_train.drop(columns='target')

        if task == 'classification':
            if model == 'logreg':
                model_train = LogisticRegression(**dict_parameters)
                model_train.fit(x, y)
                joblib.dump(model_train, f'models/{path_model}')
                return 201
            elif model == 'xgb_classifier':
                model_train = xgb.XGBClassifier(**dict_parameters)
                model_train.fit(x, y)
                joblib.dump(model_train, f'models/{path_model}')
                return 201
        elif task == 'regression':
            if model == 'linreg':
                model_train = LinearRegression(**dict_parameters)
                model_train.fit(x, y)
                joblib.dump(model_train, f'models/{path_model}')
                return 201
            elif model == 'xgb_regressor':
                model_train = xgb.XGBRegressor()(**dict_parameters)
                model_train.fit(x, y)
                joblib.dump(model_train, f'models/{path_model}')
                return 201

    @api.expect(req_delete)
    @api.response(204, description='model successfully deleted')
    def delete(self):
        """
        Delete model
        """
        request_data = req_delete.parse_args()
        path_model = request_data.get('path_model')
        error('path_model', path_model)
        os.remove(f'models/{path_model}')
        return 204


@api.route('/predict', methods=['PUT'])
@api.expect(req_post_args_file)
class ModelPredict(Resource):
    @api.doc(params={
        'file': 'Choose csv file for prediction (without target)',
        'path_model': 'Input name of your model'
    }
    )
    def put(self):
        """
        Get prediction on input csv

        """
        request_data = req_post_args_file.parse_args()
        path_model = request_data.get('path_model')
        error('path_model', path_model)
        data_test = pd.read_csv(request_data['file'])

        model_loaded = joblib.load(f'models/{path_model}')
        return model_loaded.predict(data_test).tolist()


@api.route('/available_models', methods=['GET'],
           doc={'description': 'Get available models for training'})
class GetAvailableModels(Resource):
    @staticmethod
    def get():
        """
        available model for training
        """

        return {'classification': ['logreg', 'xgb_classifier'],
                'regression': ['linreg', 'xgb_regressor']}

if __name__ == '__main__':
    app.run(debug=True)