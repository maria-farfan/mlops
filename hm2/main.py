from flask import Flask
from flask_restx import Api, Resource, reqparse, abort
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import ast
from werkzeug.datastructures import FileStorage
import glob
import pickle
from sqlalchemy import create_engine

POSTGRES_HOST = os.environ['POSTGRES_HOST']
POSTGRES_DB = os.environ['POSTGRES_DB']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
POSTGRES_CONN_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

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

req_post_args_file = reqparse.RequestParser()
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
def get_data(conn=POSTGRES_CONN_STRING):
    engine_postgres = create_engine(conn)
    data = pd.read_sql_query(
        """
        SELECT
            "col0",
            "col1",
            "target"
        FROM public.dataset;
        """,
        engine_postgres
    )
    engine_postgres.dispose()
    return data
def create_model_store(model_dir):
    files = glob.glob(f'./{model_dir}/*')
    model_store = {}
    for file in files:
        model_store[file.split('/')[-1][:-4]] = file
    return model_store

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_store = create_model_store(model_dir)
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
        'path_model': 'Unique name of your model'
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

        data_train = get_data(conn=POSTGRES_CONN_STRING)
        y, x = data_train['target'], data_train.drop(columns='target')

        if task == 'classification':
            if model == 'logreg':
                model_train = LogisticRegression(**dict_parameters)
                model_train.fit(x, y)
                with open(os.path.join(model_dir, f'{path_model}.pkl'), 'wb') as f:
                    pickle.dump(model_train, f)
                model_store[path_model] = os.path.join(model_dir, f'{path_model}.pkl')
                return 201
            elif model == 'xgb_classifier':
                model_train = xgb.XGBClassifier(**dict_parameters)
                model_train.fit(x, y)
                with open(os.path.join(model_dir, f'{path_model}.pkl'), 'wb') as f:
                    pickle.dump(model_train, f)
                model_store[path_model] = os.path.join(model_dir, f'{path_model}.pkl')
                return 201
        elif task == 'regression':
            if model == 'linreg':
                model_train = LinearRegression(**dict_parameters)
                model_train.fit(x, y)
                with open(os.path.join(model_dir, f'{path_model}.pkl'), 'wb') as f:
                    pickle.dump(model_train, f)
                model_store[path_model] = os.path.join(model_dir, f'{path_model}.pkl')
                return 201
            elif model == 'xgb_regressor':
                model_train = xgb.XGBRegressor()(**dict_parameters)
                model_train.fit(x, y)
                with open(os.path.join(model_dir, f'{path_model}.pkl'), 'wb') as f:
                    pickle.dump(model_train, f)
                model_store[path_model] = os.path.join(model_dir, f'{path_model}.pkl')
                return 201

    @api.expect(req_delete)
    @api.response(204, description='model successfully deleted')
    def delete(self):
        """
        Delete model
        """
        request_data = req_delete.parse_args()
        path_model = request_data.get('path_model')
        try:
            model_path = model_store[path_model]
            os.remove(model_path)
            del model_store[path_model]
        except KeyError:
            raise KeyError('Model not found')
        return 204


@api.route('/predict', methods=['PUT'])
@api.expect(req_post_args_file)
class ModelPredict(Resource):
    @api.doc(params={
        'path_model': 'Input name of your model'
    }
    )
    def put(self):
        """
        Get prediction on input csv

        """
        request_data = req_post_args_file.parse_args()
        path_model = request_data.get('path_model')
        model_path = model_store[path_model]
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        data_test = get_data(conn=POSTGRES_CONN_STRING)
        return model.predict(data_test).tolist()


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
    app.run(debug=True, port=5000, host='0.0.0.0')