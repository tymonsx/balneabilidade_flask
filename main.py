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
    return "Hello World from Flask"
    
@app.route('/todosResultados', methods=['GET'])
def retornaTodosResultados():
    cidade = request.args.get('cidade')
    praia = request.args.get('praia')
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    print(dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())])
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())]
    conversaoEmLista = dataFrameCsv[['Date','Enterococcus']].to_numpy().tolist()
    response = app.response_class(
        response=json.dumps(conversaoEmLista),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/resultadosUltimosDoisAnos', methods=['GET'])
def resultadosUltimosDoisAnos():
    cidade = request.args.get('cidade')
    praia = request.args.get('praia')
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    print(dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())])
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())].tail(104)
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
    print(f'cidade:{cidade.upper()}')
    print(f'praia:{praia.upper()}')
    model_path = f'modelos/model{cidade.upper()}-{praia.upper()}.h5'
    print(model_path)
    loaded = tf.keras.models.load_model(model_path)
    print(loaded.summary())
    dataFrameCsv = pandas.read_csv('sp_beaches_update.csv')
    frequenciasCsv = pandas.read_csv('frequencia_praias.csv')
    frequenciasCsv = frequenciasCsv[(frequenciasCsv["City"] == cidade.upper()) & (frequenciasCsv["Beach"] == praia.upper())]
    freqPraias = numpy.ravel(frequenciasCsv[['Frequency']])
    if freqPraias == "MENSAL":
       numMed = 4
       dataInicial = "11/02/2020"
       index_date = pandas.bdate_range(dataInicial, periods = int(numPredicoes), freq ='MS')
       datasStr=numpy.datetime_as_string(index_date, unit='D')
    elif freqPraias == "SEMANAL":
       numMed = 16
       dataInicial = date.today() 
       index_date = pandas.bdate_range(dataInicial, periods = int(numPredicoes), freq = 'C', weekmask='Mon')
       datasStr=numpy.datetime_as_string(index_date, unit='D')   
    
    print(numMed)
    dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())].tail(numMed)
    ultimasMedicoes = dataFrameCsv[['Enterococcus']].to_numpy()
    
    arrMed = ultimasMedicoes
    ##arrRes = numpy.array([],dtype=int)
    arrRes = numpy.array([])
    
    for cont in range(int(numPredicoes)):
        arrMedFormatado = arrMed.reshape((1, numMed, 1))
        predicaoProximaSemana = round(loaded.predict(arrMedFormatado)[0,0])
        arrRes = numpy.append(arrRes, predicaoProximaSemana)
        arrMed = numpy.append(arrMed, predicaoProximaSemana)
        arrMed = arrMed[-numMed:]
    
    ##print(arrRes)
    
    ##hoje = date.today() 
    ##index_date = pandas.bdate_range(hoje, periods = int(numPredicoes), freq = 'C', weekmask='Mon')
    ##datasStr=numpy.datetime_as_string(index_date, unit='D')
    
    ##print(datasStr)
    
    arrRes = numpy.clip(arrRes,0, None)
    
    arrRes = numpy.array(arrRes,dtype=int)
    
    arrRes = numpy.column_stack((datasStr, arrRes))
    
    
    
    print(arrRes)
    
    response = app.response_class(
        response=json.dumps(arrRes.tolist()),
        status=200,
        mimetype='application/json'
    )
    return response
    
##@app.route('/previsaoProximaSemana', methods=['GET'])
##def preveProximaSemana():
    ##cidade = request.args.get('cidade')
    ##praia = request.args.get('praia')
    ##loaded = tf.keras.models.load_model(f'modelos/model{cidade}-{praia}.h5')
    ##print(loaded.summary())
    ##dataFrameCsv = pandas.read_csv('resultados_mais_recentes_sp_beaches.csv')
    
    ##frequenciasCsv = pandas.read_csv('frequencia_praias.csv')
    ##frequenciasCsv = frequenciasCsv[(frequenciasCsv["City"] == cidade.upper()) & (frequenciasCsv["Beach"] == praia.upper())]
    ##freqPraias = numpy.ravel(frequenciasCsv[['Frequency']])
    ##if freqPraias == "MENSAL":
        ##numMed = 4
    ##elif freqPraias == "SEMANAL":
       ##numMed = 16
       
    ##dataFrameCsv = dataFrameCsv[(dataFrameCsv["City"] == cidade.upper()) & (dataFrameCsv["Beach"] == praia.upper())].tail(numMed)
    ##ultimasMedicoes = dataFrameCsv[['Enterococcus']].to_numpy()
    ##ultimasMedicoes = ultimasMedicoes.reshape((ultimasMedicoes.shape[1], ultimasMedicoes.shape[0], 1))
    ##print(ultimasMedicoes)
    ##predicaoProximaSemana = loaded.predict(ultimasMedicoes)
    ##response = app.response_class(
        ##response=json.dumps(round(predicaoProximaSemana[0,0])),
        ##status=200,
        ##mimetype='application/json'
    ##)
    ##return response

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)