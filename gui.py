import os
import video_super_resolver as vsr
import PySimpleGUI as sg

sg.theme('DarkAmber')   

layout = [
            [sg.Text('Path to Video')],
            [sg.Input(), sg.FileBrowse()],
            [sg.Text('Path to Output Folder')],
            [sg.Input(), sg.FolderBrowse()],
            [sg.Button('Ok'), sg.Button('Cancel')] 
         ]


window = sg.Window('VideoSRGAN', layout)

while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
        
    elif event == 'OK' and values[0] == '' or values[0] == '':
        sg.Popup('Select valid paths!')

    ippath = values[0]
    oppath = values[1]
    vsr.evaluate(ippath, oppath)
    sg.Popup('Video Enhanced!')
    
    


window.close()



