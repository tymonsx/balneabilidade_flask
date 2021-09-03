from flask import Flask, escape, request, jsonify, json
from markupsafe import escape
import pandas as pandas
from flask_cors import CORS, cross_origin
from datetime import date
import numpy as numpy
import tensorflow as tf
 
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
 
@app.route("/")
def hello():
    return "Webservice do App Balneabilidade"
 
@app.route('/todosResultados', methods=['GET'])
def retornaTodosResultados():
    cidade = request.args.get('cidade')
    praia = request.args.get('praia')
 
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())]
 
    conversaoEmLista = dataFrameCsv[['Date','Enterococcus']].to_numpy().tolist()
    response = app.response_class(
        response=json.dumps(conversaoEmLista),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/historicoMedicoes', methods=['GET'])
def montarTabelaHistorico():
    cidade = request.args.get('cidade')
    praia = request.args.get('praia')
 
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())].tail(10)
 
    conversaoEmLista = dataFrameCsv[['Date','Enterococcus']].to_numpy().tolist()
    response = app.response_class(
        response=json.dumps(conversaoEmLista),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/montarGrafico', methods=['GET'])
def montarGraficoLimiteDatas():
    cidade = request.args.get('cidade')
    praia = request.args.get('praia')
    inicio = request.args.get('inicio')
    fim = request.args.get('fim')
 
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper()) & (dataFrameCsv["Date"] >= inicio) & (dataFrameCsv["Date"] <= fim)]

    conversaoEmLista = dataFrameCsv[['Date','Enterococcus']].to_numpy().tolist()
    response = app.response_class(
        response=json.dumps(conversaoEmLista),
        status=200,
        mimetype='application/json'
    )
    return response
 
@app.route('/previsaoProximasSemanas', methods=['GET'])
def preveProximasSemanas():
    cidade = request.args.get('cidade')
    praia = request.args.get('praia')
    numPredicoes = request.args.get('numPredicoes')
 
    model_path = f'modelos/model{cidade.upper()}-{praia.upper()}.h5'
    loaded = tf.keras.models.load_model(model_path)
 
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    frequenciasCsv = pandas.read_csv('frequencia_praias.csv')
    frequenciasCsv = frequenciasCsv[(frequenciasCsv["City"] == cidade.upper()) & (frequenciasCsv["Beach"] == praia.upper())]
    freqPraias = numpy.ravel(frequenciasCsv[['Frequency']])
 
    if freqPraias == "MENSAL":
        numMed = 4
        dataInicial = date.today()
        index_date = pandas.bdate_range(dataInicial, periods = int(numPredicoes), freq ='MS')
    elif freqPraias == "SEMANAL":
        numMed = 16
        dataInicial = date.today() 
        index_date = pandas.bdate_range(dataInicial, periods = int(numPredicoes), freq = 'C', weekmask='Mon')
    
    datasStr=numpy.datetime_as_string(index_date, unit='D')
 
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())].tail(numMed)
    ultimasMedicoes = dataFrameCsv[['Enterococcus']].to_numpy()
 
    arrRes = numpy.array([])
 
    for cont in range(int(numPredicoes)):
        ultimasMedicoes = ultimasMedicoes.reshape((1, numMed, 1))
        predicaoProximaSemana = round(loaded.predict(ultimasMedicoes)[0,0])
        arrRes = numpy.append(arrRes, predicaoProximaSemana)
        ultimasMedicoes = numpy.append(ultimasMedicoes, predicaoProximaSemana)
        ultimasMedicoes = ultimasMedicoes[-numMed:]
 
    arrRes = numpy.clip(arrRes,0, None)
    arrRes = numpy.array(arrRes,dtype=int)
    arrRes = numpy.column_stack((datasStr, arrRes))
 
    response = app.response_class(
        response=json.dumps(arrRes.tolist()),
        status=200,
        mimetype='application/json'
    )
    return response
 
