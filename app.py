# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:35:16 2022

@author: abhinav.kumar
"""
import dash
from dash.exceptions import PreventUpdate
from dash import Dash, html, dcc, Input, Output, State

import dash_bootstrap_components as dbc

import io
import base64

import pandas as pd
import datetime

from python_speech_features import mfcc
import librosa as lb
import pickle
import os
import numpy as np
from tensorflow.keras.models import load_model




model = load_model('assets/model/conv.h5')

class config2:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=8000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.rate=rate
        self.step = int(rate/1)
        self.model_path = os.path.join('../working/models', mode+'.h5')
        self.p_path = os.path.join('../working/pickle', mode+'.p')

config = config2()        

with open("assets/pickle/conv3.p", 'rb') as handle:
    config = pickle.load(handle)


def building_model(filePath):
    X=[]
    min_, max_ = config.min, config.max
    signal, rate = lb.load(filePath, sr=8000) #downSampling
    signal, _ = lb.effects.trim(signal, top_db=30)
    for i in range(1, int(signal.shape[0]/rate)):
        sample = signal[(i-1)*config.step:i*config.step]
        try:
            X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt,
                        nfft=config.nfft)
        except:
            continue
        X.append(X_sample)
    X = np.array(X)
    X = (X - min_)/(max_ - min_)
#     print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decode = base64.b64decode(content_string)
    X = building_model(io.BytesIO(decode))
    pred = model.predict(X)
    a = pd.DataFrame([config.classes[i] for i in pred.argmax(axis=1)]).value_counts(ascending=False).to_frame().index.to_list()
    mapping = {'tired':'Overtired', 'burping':'Burping', 'discomfort':'Discomfort', 'belly_pain':'Tummy Pain', 'hungry':"Hungry"}
    if len(a)>1:
        result = ", ".join([i[0] for i in a])
        result = result.upper()
    else:
        result = ", ".join([i[0] for i in a])
        result = mapping[result].upper()
    
    return html.Div([
        html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),
        html.Audio(src=contents, controls=True),
        html.Hr(),
        html.Div('Final Result'),
        html.Pre(result ,className='model_result')
    ], style={'text-align':'center'})


def parse_contents2(path):
    X = building_model(path)
    pred = model.predict(X)
    a = pd.DataFrame([config.classes[i] for i in pred.argmax(axis=1)]).value_counts(ascending=False).to_frame().index.to_list()
#     result = ", ".join([i[0] for i in a])
    result = a[0][0]
    mapping = {'tired':'Overtired', 'burping':'Burping', 'discomfort':'Discomfort', 'belly_pain':'Tummy Pain', 'hungry':"Hungry"}
    return html.Div([
        html.Pre(mapping[result], className='model_result')
    ], style={'text-align':'center'}), mapping[result]




d = pd.read_csv('assets/data.csv')

#style
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Roboto&display=swap"
]

# creating app
app = Dash(__name__, external_stylesheets=external_stylesheets,
                   suppress_callback_exceptions = False,
                  meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])

app.title = "Crying Analytics: Understand Your Baby Cry!"

predict = html.Div([
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '1rem 1rem 1rem 1rem'
            },
            multiple=False
        ),
    ]),
    html.Div([
        html.Div(id='output-data-upload')
    ], style={'margin':'1rem'})
], className="")


competition = html.Div([
    html.Div([
        html.P('', id='store', hidden=True),
        dbc.Button(
            html.I("i"),
            id="positioned-toast-toggle",
            className='m-1 btn_color_1 text-color',
            style={'width':'20px', 'border-radius':'20px', 'padding':'0'}
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Instructions"),
                dbc.PopoverBody(
                    html.Div([
                        html.P('You will be provided 5 different audio clip.'),
                        html.P('After listening clip you have 5 options, you\
                               can only choose one.'),
                        html.P('Click on Check Button then. This will trigger\
                               AI to predict the audio.'),
                        html.P('Orginal Anwser will also highlighted.'),
                        html.P('If your had choosen correct anwser then your\
                               score will update. Same goes for AI.'),
                        html.P("That's All!!! Enjoy.")
                    ])
                ),
            ],
            target="positioned-toast-toggle",
            trigger="click",
        ),
    ]),
    html.Div([
        html.Div([
            html.H4('Your Score : 0', id='your-score')
        ], className='options'),
        html.Div([
            html.H4('AI Score : 0', id='model-score')
        ], className='options')
    ],className="competition-container"),
    html.Div([
            html.H5('Q - 1/5', id='question-number'),
            html.Audio(src='assets/discomfort-1.wav',id='audio-play',
                       controls=True)
        ], className="", style={'text-align':'center'}),
    html.Div([
        html.Div([
                html.H5('Your Prediction:-'),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem("Hungry", id="button-item-1",
                                          n_clicks=0, action=True,
                                          active=False,
                                          class_name='list-option'),
                        dbc.ListGroupItem("Discomfort", id="button-item-2",
                                          n_clicks=0, action=True,
                                          active=False,
                                          class_name='list-option'),
                        dbc.ListGroupItem("Burping", id="button-item-3",
                                          n_clicks=0, action=True,
                                          active=False,
                                          class_name='list-option'),
                        dbc.ListGroupItem("Overtired", id="button-item-4",
                                          n_clicks=0, action=True,
                                          active=False,
                                          class_name='list-option'),
                        dbc.ListGroupItem("Tummy Pain", id="button-item-5",
                                          n_clicks=0, action=True,
                                          active=False, 
                                          class_name='list-option'),
                    ],class_name="container-color"
                ),
        html.P(id="counter"),
        ], className="options"),#options d-flex flex-column align-items-center
        html.Div([
            html.H5('AI Prediction:-'),
            html.Div([
                    html.P('')
                ], id='model-pred')
        ], className="options"),
    ], className='competition-container'),
    html.Div([
        html.Div([
            dbc.Button('Check', id='check_answer',
                       class_name="btn_color_1 text-color"),
            dbc.Toast(
                [html.P("You have to select one option. There is no negative\
                        marking!!!", className="mb-0")],
                id="auto-toast",
                icon="info",
                duration=4000,
                is_open=False,
                style={"position": "fixed", "top": 66, "right": 10,
                       "width": 350},
            ),
        ],className='w-50'),
        html.Div([
            dbc.Button('Next', id="next_question",disabled=True,
                       style={'margin-left':'0.5rem'},
                       class_name="btn_color_1 text-color"),
            dbc.Button('Challenge Again!!', id="new_try",disabled=True,
                       class_name='',
                       style={'margin-left':'0.5rem',
                              'background-color': 'rgba(0,0,0,0)',
                              'border':'0',
                             'color':'#F0C1A2'
                             })
        ],className='w-50')
        
    ], style={'text-align':'center', 'margin-bottom':'1rem'},
                className='d-flex'),
    html.Div([
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("H", id = 'model-header'), 
                                close_button=True),
                dbc.ModalBody("", id='model-body'),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-centered",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-centered",
            centered=True,
            is_open=False,
        ),
    ])
])



app.layout = html.Div([
                html.Div([
                    html.H3("Baby's Crying Prediction", id='welcome_user',
                            className="header-title text-color"),
                    html.P(["Most of us don't know what's a baby going through \
                            when they cry. Either your little cutie pie is \
                    HUNGRY or having TUMMY PAIN or due to  DISCOMFORT or \
                        BURPING or OVERTIRED?"], className='header-description \
                            text-color'),
                    html.P("Let's take a help of Machine Learning Model\
                    and predict this intricate crying, so you can take \
                        imperative action.",
                        className='header-description text-color')
                ], className = 'header-container container-color'),
                dcc.Store(id='question-index'),
                dcc.Store(id='question-index2'),
                html.Div([
                    dbc.RadioItems(
                        id='time_select',
                        
                        className="btn-group d-flex justify-content-around text-color",
                        inputClassName="btn-check",
                        labelClassName="btn_color_1 text-color btn-border",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Predict with AI", "value": 1},
                            {"label": "Compete with AI", "value": 2},
                        ],
                        value=0,
                    ),
                ],className='radio-container container-color'),
                html.Div([
                    'data2'
                ],id='insert_page2',
                    className='data-container container-color flex-grow-1 text-color')
            ],className='big-container')


@app.callback([Output("next_question",'children')],
              Input("question-number", 'children'),
              prevent_initial_call=True)
def change_next_text(q1):
    q_num = int(q1.split('-')[-1].split("/")[0].strip())
    if q_num == 5:
        return ["Finish"]
    else:
        return ['Next']
    
@app.callback(
    [Output("modal-centered", "is_open"), Output('model-body', 'children'),
     Output('model-header','children')],
    [Input("next_question", "n_clicks"), Input("close-centered", "n_clicks")],
    [State("modal-centered", "is_open"), State('next_question', 'children'),
     State("model-score", 'children'), State('your-score', 'children')],
)
def toggle_modal(n1, n2, is_open, finish, model, your):
    model_score = int(model.split(':')[-1].split("/")[0].strip())
    your_score = int(your.split(':')[-1].split("/")[0].strip())
    if your_score > model_score:
        text = 'You Won'
        text2 = "It's feel nice to beat AI, Right?"
    elif model_score > your_score:
        text = "You Lost"
        text2 = "Take Another challenge and beat this AI."
    elif your_score == 5:
        text = "That's Draw"
        text2 = "Nice that's Draw but you got good fight here with AI. COOL"
    else:
        text = "That's Draw"
        text2 = "Nice that's Draw but take another challenge and beat AI."
    btn_click = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if btn_click == 'next_question':
        if finish=="Finish":
            return [not is_open, text2, text]
    elif n2:
        return [not is_open, '', '']
    raise PreventUpdate
    
@app.callback(
    [Output("model-pred", 'children'), Output("model-score", 'children')],
    [Input("check_answer", 'n_clicks'),Input("next_question", 'n_clicks'),
     Input("new_try", 'n_clicks')],
    [State("question-index", 'data'), State("question-number", 'children'),
     State("button-item-1",'active'), State("button-item-2",'active'),
     State("button-item-3",'active'), State("button-item-4",'active'),
     State("button-item-5",'active'), State("model-score", 'children'),
     State("question-index2", 'data')], prevent_initial_call=True)
def model_pred(n1,n2,n3, data1, q1, a1, a2, a3, a4, a5, s2, data2):
    time1 = data1.split(", ")[-1]
    try:
        time2 = data2.split(", ")[-1]
        if time2> time1:
            q_index_orgi = data2
        else:
            q_index_orgi = data1
    except:
        q_index_orgi = data1
    q_num = int(q1.split('-')[-1].split("/")[0].strip())
    model_score = int(s2.split(':')[-1].split("/")[0].strip())
    q_index=q_index_orgi.split(',')
    q_index=int(q_index[q_num-1])
    anw_true = d.iloc[q_index, 2]
    label = ['Hungry', 'Discomfort', 'Burping', 'Overtired', 'Tummy Pain']
    b1 = [a1, a2, a3, a4, a5]
    try:
        b = [i  for i in range(len(b1)) if b1[i] == True][0]
    except:
        b = 'not_selected'
    btn_click = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if b != 'not_selected':
        if btn_click == 'check_answer':
            path = d.iloc[q_index, 0]
            html_part, x = parse_contents2(path)
            if label[anw_true] == x:
                model_score += 1
            s2 = 'AI Score : '+str(model_score)
            return [html_part, s2]
        elif btn_click == 'next_question':
            return [html.P(''), s2]
    elif btn_click == 'new_try':
        return [html.P(''), "AI Score : 0"]
    else:
        raise PreventUpdate

        
        

@app.callback(
    [Output("button-item-1",'active'),Output("button-item-2",'active'),
     Output("button-item-3",'active'),Output("button-item-4",'active'),
     Output("button-item-5",'active'),Output("button-item-1",'color'),
     Output("button-item-2",'color'),Output("button-item-3",'color'),
     Output("button-item-4",'color'),Output("button-item-5",'color'),
     Output("button-item-1",'disabled'),Output("button-item-2",'disabled'),
     Output("button-item-3",'disabled'),Output("button-item-4",'disabled'),
     Output("button-item-5",'disabled'),Output("check_answer", 'disabled'),
     Output("next_question", 'disabled'),Output("auto-toast", "is_open"),
     Output("your-score", 'children'),Output("question-number", 'children'),
     Output('new_try', 'style'), Output('new_try', 'disabled'),
     Output('audio-play', 'src')], 
    [Input("button-item-1",'n_clicks'),Input("button-item-2",'n_clicks'),
     Input("button-item-3",'n_clicks'),Input("button-item-4",'n_clicks'),
     Input("button-item-5",'n_clicks'),Input("check_answer", 'n_clicks'),
     Input("next_question", 'n_clicks'),Input("new_try", 'n_clicks'),
     Input("question-index", 'data')],
    [State("button-item-1",'active'),State("button-item-2",'active'),
     State("button-item-3",'active'),State("button-item-4",'active'),
     State("button-item-5",'active'),State("your-score", 'children'),
     State("question-number", 'children'), State("next_question", 'children'),
     State('new_try', 'style'), State('audio-play', 'src'),
     State("question-index2", 'data'),State("check_answer", 'disabled')],
     prevent_initial_call=True)
def activate_btn(n1, n2, n3, n4, n5, check,next_q,new_try,data1, a1, a2, a3,
                 a4, a5, s1, q1, finish, style, src, data2, isdisable):
    
    time1 = data1.split(", ")[-1]
    try:
        time2 = data2.split(", ")[-1]
        if time2> time1:
            q_index_orgi = data2
        else:
            q_index_orgi = data1
    except:
        q_index_orgi = data1
    your_score = int(s1.split(':')[-1].split("/")[0].strip())
    q_num = int(q1.split('-')[-1].split("/")[0].strip())
    btn_click = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    q_index=q_index_orgi.split(',')
    q_index=int(q_index[q_num-1])
    anw_true = d.iloc[q_index, 2]
    
    if btn_click == "check_answer":
        b1 = [a1, a2, a3, a4, a5]
        try:
            b = [i  for i in range(len(b1)) if b1[i] == True][0]
        except:
            b = 'not_selected'
        if b == 'not_selected':
            return [False, False, False, False, False, 0,0,0,0,0, False, False,
                    False, False, False]+[False]+[True]+[True]+[s1]+[q1]+[style]+[True]+[src]
        else:
            label = ['Hungry', 'Discomfort', 'Burping', 'Overtired',
                     
                     'Tummy Pain']
            col = [0,0,0,0,0]
            print('anw-', label[anw_true], '  b-', label[b])
            if anw_true == b:
                col[b] = 'success'
                your_score += 1
            else:
                col[b] = '#B8625E'
                col[anw_true] = '#585C42'
            a = [False, False, False, False, False]
            a[b] = True
            a[anw_true] = True
            c = [True, True, True, True, True]
            c[b]= False
            c[anw_true]=False
            s1 = 'Your Score : '+str(your_score)
            return a+col+c+[True]+[False]+[False]+[s1]+[q1]+[style]+[False]+[src]
    elif btn_click == "next_question":
        if finish != "Finish":
            q_index=q_index_orgi.split(',')
            q_index=int(q_index[q_num])
            path = d.iloc[q_index, 0]
            q1 = "Q - "+str(q_num+1)+"/5"
            return [False, False, False, False, False, "","","","","", False,
                    False, False, False, False]+[False]+[True]+[False]+[s1]+[q1]+[style]+[True]+[path]
        else:
            style={'margin-left':'2rem', 'background-color': 'rgba(0,0,0,0.1)',
                   'color':'black' }
            return [False, False, False, False, False, "","","","","", True,
                    True, True, True, True]+[True]+[True]+[False]+[s1]+[q1]+[style]+[False]+[src]
                    
    elif len(btn_click.split("-")[-1]) == 1:
        print('disable - ', isdisable)
        print(n1, n2, n3, n4, n5)
        if isdisable:
            raise PreventUpdate
        else:
            a = [False, False, False, False, False]
            a[int(btn_click.split("-")[-1])-1] = True
            return a+[n1, n2, n3, n4, n5]+[False, False, False, False, False]+[False]+[True]+[False]+[s1]+[q1]+[style]+[True]+[src]
    
    elif btn_click == 'question-index':
        q_index=q_index_orgi.split(',')
        q_index=int(q_index[0])
        path = d.iloc[q_index, 0]
        return[False, False, False, False, False, "","","","","", False, False,
               False, False, False]+[False]+[True]+[False]+[s1]+[q1]+[style]+[True]+[path]
        
    elif btn_click == 'new_try':
        text = ", ".join([str(i) for i in list(d.sample(5).index)]+
                         [str(datetime.datetime.now())])
        q_index=text.split(',')
        q_index=int(q_index[0])
        path = d.iloc[q_index, 0]
        q1 = "Q - 1/5"
        s1 = 'Your Score : 0'
        style=style={'margin-left':'0.5rem', 
                     'background-color': 'rgba(0,0,0,0)', 'border':'0',
                             'color':'#F0C1A2'}
        return[False, False, False, False, False, "","","","","", False, False,
               False, False, False]+[False]+[True]+[False]+[s1]+[q1]+[style]+[True]+[path]
    else:
        PreventUpdate()


@app.callback([Output("question-index2", 'data')],
              [Input("new_try", 'n_clicks')],
              prevent_initial_call=True)
def new_questions(content):    
    text = ", ".join([str(i) for i in list(d.sample(5).index)]+
                     [str(datetime.datetime.now())])
    if content:
        return [text]


@app.callback(
    [Output('insert_page2', 'children'), Output('question-index', 'data')],
    [Input('time_select', 'value')],
    prevent_initial_call=False)
def update_output(content):    
    print(content)
    if content == 1:
        return [predict, ""]
    elif content == 2:
        text = ", ".join([str(i) for i in list(d.sample(5).index)]+
                         [str(datetime.datetime.now())])
        return [competition, text]
    else:
        return [html.Div([
            html.H1('Select Any one from above two options')
        ], style={'text-align':'center', 'padding':'5rem'}), ""]
    
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)
def predict_using_model(content, name, date):
    if content is not None:
        children = [
            parse_contents(content, name, date)]
        return children
                           
                           
                           
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)


