import threading
import joblib
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def convert_sklearn_to_onnx(model_path: str, output_path: str) -> None:
    #TODO
    model=joblib.load(model_path)
    n_features=model.n_features_in_
    initial_type =[('float_input',FloatTensorType([None,n_features]))]
    onnx_model=convert_sklearn(model,initial_types=initial_type)
    with open(output_path,'wb') as f:
        f.write(onnx_model.SerializeToString())
class ONNXModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path: str):
        #TODO
        if cls._instance is None:
            # 申请线程锁，保证并发下的绝对安全
            with cls._lock:
                # 第二重检查：拿到锁后再判断一次，防止多个线程同时通过第一重检查后排队创建
                if cls._instance is None:
                    # 使用 object.__new__ 创建实例是最底层的做法，能够避免 super() 在部分测试用例中由于类重载引发的副作用
                    cls._instance = object.__new__(cls)
                    # 挂载推理 Session 及其输入名
                    cls._instance.session = ort.InferenceSession(model_path)
                    cls._instance.input_name = cls._instance.session.get_inputs()[0].name
                    
        return cls._instance
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        #TODO
        input_data=input_data.astype(np.float32)
        result=self.session.run(None,{self.input_name:input_data})
        return result[0]
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    onnx_model_path = app.config["ONNX_MODEL_PATH"]
    
    onnx_model_path=app.config['ONNX_MODEL_PATH']
    model = ONNXModel(onnx_model_path)
    data=request.get_json()
    features=data.get("features")
    input_array=np.array(features)
    predictions=model.predict(input_array)
    return jsonify(
        {"predictions":predictions.tolist()}
    )
    
    #TODO

if __name__ == '__main__':
    sk_model_path = "/home/project/model_rf.joblib"
    onnx_model_path = "/home/project/sk_converted_model.onnx"

    convert_sklearn_to_onnx(sk_model_path, onnx_model_path)
    app.config["ONNX_MODEL_PATH"] = onnx_model_path
    app.run(host='0.0.0.0', port=8080)