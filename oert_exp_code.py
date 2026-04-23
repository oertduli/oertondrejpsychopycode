#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on April 23, 2026, at 13:31
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'ver3OertOndrej_PIT_experiment_psychopy'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [800, 600]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Julia\\Desktop\\oert_exp_code.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=-1,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='event'
        )
    if deviceManager.getDevice('welcomeKey') is None:
        # initialise welcomeKey
        welcomeKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcomeKey',
        )
    if deviceManager.getDevice('payKey') is None:
        # initialise payKey
        payKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='payKey',
        )
    if deviceManager.getDevice('breakKey') is None:
        # initialise breakKey
        breakKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='breakKey',
        )
    if deviceManager.getDevice('endKey') is None:
        # initialise endKey
        endKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='endKey',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='Pyglet',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='Pyglet'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    # Run 'Begin Experiment' code from conditionCode
    import random
    
    orderNum = random.randint(1, 6)
    scheduleNum = random.randint(1, 20)
    
    scheduleFile = f"PIT_experiment_schedules/order_{orderNum}/schedule_{scheduleNum}.csv"
    
    expInfo['orderNum'] = orderNum
    expInfo['scheduleNum'] = scheduleNum
    
    print("Chosen order:", orderNum)
    print("Chosen schedule:", scheduleNum)
    print("Chosen file:", scheduleFile)
    bgImage = visual.ImageStim(
        win=win,
        name='bgImage', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    balloonImageStart = visual.ImageStim(
        win=win,
        name='balloonImageStart', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    welcomeText = visual.TextStim(win=win, name='welcomeText',
        text='Welcome to the task\n\nPress space to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    welcomeKey = keyboard.Keyboard(deviceName='welcomeKey')
    curvedLine = visual.ImageStim(
        win=win,
        name='curvedLine', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    
    # --- Initialize components for Routine "start" ---
    bgImage_2 = visual.ImageStim(
        win=win,
        name='bgImage_2', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    balloonImage = visual.ImageStim(
        win=win,
        name='balloonImage', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    moneyImage = visual.ImageStim(
        win=win,
        name='moneyImage', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyText = visual.TextStim(win=win, name='moneyText',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    startText = visual.TextStim(win=win, name='startText',
        text='Start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    curvedLine_3 = visual.ImageStim(
        win=win,
        name='curvedLine_3', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    
    # --- Initialize components for Routine "indicators" ---
    bgImage_3 = visual.ImageStim(
        win=win,
        name='bgImage_3', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    balloonImage_2 = visual.ImageStim(
        win=win,
        name='balloonImage_2', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyImage_2 = visual.ImageStim(
        win=win,
        name='moneyImage_2', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    moneyText_2 = visual.TextStim(win=win, name='moneyText_2',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    indicatorTitle = visual.TextStim(win=win, name='indicatorTitle',
        text='Warning: incoming airships..... \nPay 50 to reveal more?',
        font='Arial',
        pos=(0, 0.58), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    indicatorInfo = visual.TextStim(win=win, name='indicatorInfo',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=0.0, 
        languageStyle='LTR',
        depth=-6.0);
    indicatorImage = visual.ImageStim(
        win=win,
        name='indicatorImage', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.4, 0.35), draggable=False, size=(0.16, 0.16),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    curvedLine_4 = visual.ImageStim(
        win=win,
        name='curvedLine_4', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    
    # --- Initialize components for Routine "pay_decision" ---
    bgImage_4 = visual.ImageStim(
        win=win,
        name='bgImage_4', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    balloonImage_3 = visual.ImageStim(
        win=win,
        name='balloonImage_3', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyImage_3 = visual.ImageStim(
        win=win,
        name='moneyImage_3', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    moneyText_3 = visual.TextStim(win=win, name='moneyText_3',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    decisionText = visual.TextStim(win=win, name='decisionText',
        text='Payment option\n\nPress Y for yes\nPress N for no',
        font='Arial',
        pos=(0, 0.30), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    payKey = keyboard.Keyboard(deviceName='payKey')
    curvedLine_5 = visual.ImageStim(
        win=win,
        name='curvedLine_5', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    
    # --- Initialize components for Routine "shield_phase" ---
    bgImage_5 = visual.ImageStim(
        win=win,
        name='bgImage_5', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    balloonImage_4 = visual.ImageStim(
        win=win,
        name='balloonImage_4', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    moneyImage_4 = visual.ImageStim(
        win=win,
        name='moneyImage_4', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyText_4 = visual.TextStim(win=win, name='moneyText_4',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    shieldInfo = visual.TextStim(win=win, name='shieldInfo',
        text='',
        font='Arial',
        pos=(0.18, -0.48), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    shieldCondition = visual.TextStim(win=win, name='shieldCondition',
        text='',
        font='Arial',
        pos=(0.18, -0.62), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    shieldImage = visual.ImageStim(
        win=win,
        name='shieldImage', 
        image='PIT_experiment_images/shield.png', mask=None, anchor='center',
        ori=0.0, pos=(18, -0.05), draggable=False, size=(0.18, 0.18),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    curvedLine_6 = visual.ImageStim(
        win=win,
        name='curvedLine_6', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    
    # --- Initialize components for Routine "airship_arrival" ---
    bgImage_6 = visual.ImageStim(
        win=win,
        name='bgImage_6', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    balloonImage_5 = visual.ImageStim(
        win=win,
        name='balloonImage_5', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyImage_5 = visual.ImageStim(
        win=win,
        name='moneyImage_5', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    moneyText_5 = visual.TextStim(win=win, name='moneyText_5',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    arrivalText = visual.TextStim(win=win, name='arrivalText',
        text='Incoming airship...\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-5.0);
    arrivalInfo = visual.TextStim(win=win, name='arrivalInfo',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-6.0);
    partialText = visual.TextStim(win=win, name='partialText',
        text='Partial Feedback',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-7.0);
    noneText = visual.TextStim(win=win, name='noneText',
        text='No Feedback',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-8.0);
    curvedLine_7 = visual.ImageStim(
        win=win,
        name='curvedLine_7', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    airshipImage = visual.ImageStim(
        win=win,
        name='airshipImage', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(0.28, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    
    # --- Initialize components for Routine "airship_attack" ---
    bgImage_7 = visual.ImageStim(
        win=win,
        name='bgImage_7', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    balloonImage_6 = visual.ImageStim(
        win=win,
        name='balloonImage_6', 
        image='PIT_experiment_images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0 ,-0.4), draggable=False, size=(0.24, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyImage_6 = visual.ImageStim(
        win=win,
        name='moneyImage_6', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    moneyText_6 = visual.TextStim(win=win, name='moneyText_6',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    attackText = visual.TextStim(win=win, name='attackText',
        text='Attack',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-5.0);
    attackInfo = visual.TextStim(win=win, name='attackInfo',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-6.0);
    partialAttackText = visual.TextStim(win=win, name='partialAttackText',
        text='Partial Feedback',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-7.0);
    noneAttackText = visual.TextStim(win=win, name='noneAttackText',
        text='No Feedback',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-8.0);
    curvedLine_8 = visual.ImageStim(
        win=win,
        name='curvedLine_8', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    airshipImage_2 = visual.ImageStim(
        win=win,
        name='airshipImage_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(0.28, 0.28),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    
    # --- Initialize components for Routine "trial_end" ---
    bgImage_8 = visual.ImageStim(
        win=win,
        name='bgImage_8', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyImage_7 = visual.ImageStim(
        win=win,
        name='moneyImage_7', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.4), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    moneyText_7 = visual.TextStim(win=win, name='moneyText_7',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    trialEndText = visual.TextStim(win=win, name='trialEndText',
        text='Trial finished',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    curvedLine_9 = visual.ImageStim(
        win=win,
        name='curvedLine_9', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    
    # --- Initialize components for Routine "break_screen" ---
    bgImage_9 = visual.ImageStim(
        win=win,
        name='bgImage_9', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    moneyImage_8 = visual.ImageStim(
        win=win,
        name='moneyImage_8', 
        image='PIT_experiment_images/money.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.6 ,-0.3), draggable=False, size=(0.10, 0.10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    moneyText_8 = visual.TextStim(win=win, name='moneyText_8',
        text='$500',
        font='Arial',
        pos=(-0.7 ,-0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    breakText = visual.TextStim(win=win, name='breakText',
        text='You may now take a short break.\n\nPress any key to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    breakKey = keyboard.Keyboard(deviceName='breakKey')
    curvedLine_10 = visual.ImageStim(
        win=win,
        name='curvedLine_10', 
        image='PIT_experiment_images/curved_line.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(1.7, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    
    # --- Initialize components for Routine "final_screen" ---
    bgImage_10 = visual.ImageStim(
        win=win,
        name='bgImage_10', 
        image='PIT_experiment_images/background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    endText = visual.TextStim(win=win, name='endText',
        text='You have completed the task.\n\nThank you for participating.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    endKey = keyboard.Keyboard(deviceName='endKey')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[bgImage, balloonImageStart, welcomeText, welcomeKey, curvedLine],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for welcomeKey
    welcomeKey.keys = ['space']
    welcomeKey.rt = []
    _welcomeKey_allKeys = []
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *bgImage* updates
        
        # if bgImage is starting this frame...
        if bgImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            bgImage.frameNStart = frameN  # exact frame index
            bgImage.tStart = t  # local t and not account for scr refresh
            bgImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bgImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bgImage.started')
            # update status
            bgImage.status = STARTED
            bgImage.setAutoDraw(True)
        
        # if bgImage is active this frame...
        if bgImage.status == STARTED:
            # update params
            pass
        
        # *balloonImageStart* updates
        
        # if balloonImageStart is starting this frame...
        if balloonImageStart.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            balloonImageStart.frameNStart = frameN  # exact frame index
            balloonImageStart.tStart = t  # local t and not account for scr refresh
            balloonImageStart.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(balloonImageStart, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'balloonImageStart.started')
            # update status
            balloonImageStart.status = STARTED
            balloonImageStart.setAutoDraw(True)
        
        # if balloonImageStart is active this frame...
        if balloonImageStart.status == STARTED:
            # update params
            pass
        
        # *welcomeText* updates
        
        # if welcomeText is starting this frame...
        if welcomeText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcomeText.frameNStart = frameN  # exact frame index
            welcomeText.tStart = t  # local t and not account for scr refresh
            welcomeText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcomeText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcomeText.started')
            # update status
            welcomeText.status = STARTED
            welcomeText.setAutoDraw(True)
        
        # if welcomeText is active this frame...
        if welcomeText.status == STARTED:
            # update params
            pass
        
        # if welcomeKey is starting this frame...
        # if welcomeKey is starting this frame...
        if welcomeKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            welcomeKey.status = STARTED
            win.callOnFlip(welcomeKey.clock.reset)
            win.callOnFlip(welcomeKey.clearEvents, eventType='keyboard')

        # check keys
        if welcomeKey.status == STARTED:
            theseKeys = welcomeKey.getKeys(keyList=['space'], waitRelease=False)
            if len(theseKeys):
                welcomeKey.keys = theseKeys[-1].name
                welcomeKey.rt = theseKeys[-1].rt
                continueRoutine = False
        
        # *curvedLine* updates
        
        # if curvedLine is starting this frame...
        if curvedLine.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            curvedLine.frameNStart = frameN  # exact frame index
            curvedLine.tStart = t  # local t and not account for scr refresh
            curvedLine.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(curvedLine, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'curvedLine.started')
            # update status
            curvedLine.status = STARTED
            curvedLine.setAutoDraw(True)
        
        # if curvedLine is active this frame...
        if curvedLine.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=welcome,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if welcomeKey.keys in ['', [], None]:  # No response was made
        welcomeKey.keys = None
    thisExp.addData('welcomeKey.keys',welcomeKey.keys)
    if welcomeKey.keys != None:  # we had a response
        thisExp.addData('welcomeKey.rt', welcomeKey.rt)
        thisExp.addData('welcomeKey.duration', welcomeKey.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trialLoop = data.TrialHandler2(
        name='trialLoop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(scheduleFile), 
        seed=None, 
    )
    thisExp.addLoop(trialLoop)  # add the loop to the experiment
    thisTrialLoop = trialLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialLoop.rgb)
    if thisTrialLoop != None:
        for paramName in thisTrialLoop:
            globals()[paramName] = thisTrialLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialLoop in trialLoop:
        trialLoop.status = STARTED
        if hasattr(thisTrialLoop, 'status'):
            thisTrialLoop.status = STARTED
        currentLoop = trialLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialLoop.rgb)
        if thisTrialLoop != None:
            for paramName in thisTrialLoop:
                globals()[paramName] = thisTrialLoop[paramName]
        
        # --- Prepare to start Routine "start" ---
        # create an object to store info about Routine start
        start = data.Routine(
            name='start',
            components=[bgImage_2, balloonImage, moneyImage, moneyText, startText, curvedLine_3],
        )
        start.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for start
        start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        start.tStart = globalClock.getTime(format='float')
        start.status = STARTED
        thisExp.addData('start.started', start.tStart)
        start.maxDuration = None
        # keep track of which components have finished
        startComponents = start.components
        for thisComponent in start.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "start" ---
        start.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_2* updates
            
            # if bgImage_2 is starting this frame...
            if bgImage_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_2.frameNStart = frameN  # exact frame index
                bgImage_2.tStart = t  # local t and not account for scr refresh
                bgImage_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_2.started')
                # update status
                bgImage_2.status = STARTED
                bgImage_2.setAutoDraw(True)
            
            # if bgImage_2 is active this frame...
            if bgImage_2.status == STARTED:
                # update params
                pass
            
            # if bgImage_2 is stopping this frame...
            if bgImage_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_2.tStop = t  # not accounting for scr refresh
                    bgImage_2.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_2.stopped')
                    # update status
                    bgImage_2.status = FINISHED
                    bgImage_2.setAutoDraw(False)
            
            # *balloonImage* updates
            
            # if balloonImage is starting this frame...
            if balloonImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloonImage.frameNStart = frameN  # exact frame index
                balloonImage.tStart = t  # local t and not account for scr refresh
                balloonImage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloonImage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloonImage.started')
                # update status
                balloonImage.status = STARTED
                balloonImage.setAutoDraw(True)
            
            # if balloonImage is active this frame...
            if balloonImage.status == STARTED:
                # update params
                pass
            
            # if balloonImage is stopping this frame...
            if balloonImage.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloonImage.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    balloonImage.tStop = t  # not accounting for scr refresh
                    balloonImage.tStopRefresh = tThisFlipGlobal  # on global time
                    balloonImage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloonImage.stopped')
                    # update status
                    balloonImage.status = FINISHED
                    balloonImage.setAutoDraw(False)
            
            # *moneyImage* updates
            
            # if moneyImage is starting this frame...
            if moneyImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage.frameNStart = frameN  # exact frame index
                moneyImage.tStart = t  # local t and not account for scr refresh
                moneyImage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage.started')
                # update status
                moneyImage.status = STARTED
                moneyImage.setAutoDraw(True)
            
            # if moneyImage is active this frame...
            if moneyImage.status == STARTED:
                # update params
                pass
            
            # if moneyImage is stopping this frame...
            if moneyImage.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage.tStop = t  # not accounting for scr refresh
                    moneyImage.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage.stopped')
                    # update status
                    moneyImage.status = FINISHED
                    moneyImage.setAutoDraw(False)
            
            # *moneyText* updates
            
            # if moneyText is starting this frame...
            if moneyText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText.frameNStart = frameN  # exact frame index
                moneyText.tStart = t  # local t and not account for scr refresh
                moneyText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText.started')
                # update status
                moneyText.status = STARTED
                moneyText.setAutoDraw(True)
            
            # if moneyText is active this frame...
            if moneyText.status == STARTED:
                # update params
                pass
            
            # if moneyText is stopping this frame...
            if moneyText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText.tStop = t  # not accounting for scr refresh
                    moneyText.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText.stopped')
                    # update status
                    moneyText.status = FINISHED
                    moneyText.setAutoDraw(False)
            
            # *startText* updates
            
            # if startText is starting this frame...
            if startText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                startText.frameNStart = frameN  # exact frame index
                startText.tStart = t  # local t and not account for scr refresh
                startText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(startText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'startText.started')
                # update status
                startText.status = STARTED
                startText.setAutoDraw(True)
            
            # if startText is active this frame...
            if startText.status == STARTED:
                # update params
                pass
            
            # if startText is stopping this frame...
            if startText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > startText.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    startText.tStop = t  # not accounting for scr refresh
                    startText.tStopRefresh = tThisFlipGlobal  # on global time
                    startText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'startText.stopped')
                    # update status
                    startText.status = FINISHED
                    startText.setAutoDraw(False)
            
            # *curvedLine_3* updates
            
            # if curvedLine_3 is starting this frame...
            if curvedLine_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_3.frameNStart = frameN  # exact frame index
                curvedLine_3.tStart = t  # local t and not account for scr refresh
                curvedLine_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_3.started')
                # update status
                curvedLine_3.status = STARTED
                curvedLine_3.setAutoDraw(True)
            
            # if curvedLine_3 is active this frame...
            if curvedLine_3.status == STARTED:
                # update params
                pass
            
            # if curvedLine_3 is stopping this frame...
            if curvedLine_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_3.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_3.tStop = t  # not accounting for scr refresh
                    curvedLine_3.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_3.stopped')
                    # update status
                    curvedLine_3.status = FINISHED
                    curvedLine_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=start,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                start.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in start.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "start" ---
        for thisComponent in start.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for start
        start.tStop = globalClock.getTime(format='float')
        start.tStopRefresh = tThisFlipGlobal
        thisExp.addData('start.stopped', start.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if start.maxDurationReached:
            routineTimer.addTime(-start.maxDuration)
        elif start.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "indicators" ---
        # create an object to store info about Routine indicators
        indicators = data.Routine(
            name='indicators',
            components=[bgImage_3, balloonImage_2, moneyImage_2, moneyText_2, indicatorTitle, indicatorInfo, indicatorImage, curvedLine_4],
        )
        indicators.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from indicatorCode
        safeCondition = "" if condition is None else str(condition)
        safeType = "" if airship_type is None else str(airship_type)
        safeColour = "" if colour is None else str(colour).strip()
        
        colourLower = safeColour.lower()
        
        indicatorInfo.setText(
            "Trial: " + str(trial_number) + "\n" +
            safeCondition + " | " +
            safeType + " | " +
            safeColour
        )
        
        indicatorFile = "PIT_experiment_images/indicator_default.png"
        airshipFile = "PIT_experiment_images/airship_default.png"
        
        if colourLower:
            indicatorFile = f"PIT_experiment_images/indicator_{colourLower}.png"
            airshipFile = f"PIT_experiment_images/airship_{colourLower}.png"
        indicatorInfo.setText('""')
        indicatorImage.setImage(indicatorFile)
        # store start times for indicators
        indicators.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        indicators.tStart = globalClock.getTime(format='float')
        indicators.status = STARTED
        thisExp.addData('indicators.started', indicators.tStart)
        indicators.maxDuration = None
        # keep track of which components have finished
        indicatorsComponents = indicators.components
        for thisComponent in indicators.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "indicators" ---
        indicators.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_3* updates
            
            # if bgImage_3 is starting this frame...
            if bgImage_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_3.frameNStart = frameN  # exact frame index
                bgImage_3.tStart = t  # local t and not account for scr refresh
                bgImage_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_3.started')
                # update status
                bgImage_3.status = STARTED
                bgImage_3.setAutoDraw(True)
            
            # if bgImage_3 is active this frame...
            if bgImage_3.status == STARTED:
                # update params
                pass
            
            # if bgImage_3 is stopping this frame...
            if bgImage_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_3.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_3.tStop = t  # not accounting for scr refresh
                    bgImage_3.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_3.stopped')
                    # update status
                    bgImage_3.status = FINISHED
                    bgImage_3.setAutoDraw(False)
            
            # *balloonImage_2* updates
            
            # if balloonImage_2 is starting this frame...
            if balloonImage_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloonImage_2.frameNStart = frameN  # exact frame index
                balloonImage_2.tStart = t  # local t and not account for scr refresh
                balloonImage_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloonImage_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloonImage_2.started')
                # update status
                balloonImage_2.status = STARTED
                balloonImage_2.setAutoDraw(True)
            
            # if balloonImage_2 is active this frame...
            if balloonImage_2.status == STARTED:
                # update params
                pass
            
            # if balloonImage_2 is stopping this frame...
            if balloonImage_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloonImage_2.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    balloonImage_2.tStop = t  # not accounting for scr refresh
                    balloonImage_2.tStopRefresh = tThisFlipGlobal  # on global time
                    balloonImage_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloonImage_2.stopped')
                    # update status
                    balloonImage_2.status = FINISHED
                    balloonImage_2.setAutoDraw(False)
            
            # *moneyImage_2* updates
            
            # if moneyImage_2 is starting this frame...
            if moneyImage_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_2.frameNStart = frameN  # exact frame index
                moneyImage_2.tStart = t  # local t and not account for scr refresh
                moneyImage_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_2.started')
                # update status
                moneyImage_2.status = STARTED
                moneyImage_2.setAutoDraw(True)
            
            # if moneyImage_2 is active this frame...
            if moneyImage_2.status == STARTED:
                # update params
                pass
            
            # if moneyImage_2 is stopping this frame...
            if moneyImage_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage_2.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage_2.tStop = t  # not accounting for scr refresh
                    moneyImage_2.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage_2.stopped')
                    # update status
                    moneyImage_2.status = FINISHED
                    moneyImage_2.setAutoDraw(False)
            
            # *moneyText_2* updates
            
            # if moneyText_2 is starting this frame...
            if moneyText_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_2.frameNStart = frameN  # exact frame index
                moneyText_2.tStart = t  # local t and not account for scr refresh
                moneyText_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_2.started')
                # update status
                moneyText_2.status = STARTED
                moneyText_2.setAutoDraw(True)
            
            # if moneyText_2 is active this frame...
            if moneyText_2.status == STARTED:
                # update params
                pass
            
            # if moneyText_2 is stopping this frame...
            if moneyText_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText_2.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText_2.tStop = t  # not accounting for scr refresh
                    moneyText_2.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText_2.stopped')
                    # update status
                    moneyText_2.status = FINISHED
                    moneyText_2.setAutoDraw(False)
            
            # *indicatorTitle* updates
            
            # if indicatorTitle is starting this frame...
            if indicatorTitle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                indicatorTitle.frameNStart = frameN  # exact frame index
                indicatorTitle.tStart = t  # local t and not account for scr refresh
                indicatorTitle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(indicatorTitle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'indicatorTitle.started')
                # update status
                indicatorTitle.status = STARTED
                indicatorTitle.setAutoDraw(True)
            
            # if indicatorTitle is active this frame...
            if indicatorTitle.status == STARTED:
                # update params
                pass
            
            # if indicatorTitle is stopping this frame...
            if indicatorTitle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > indicatorTitle.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    indicatorTitle.tStop = t  # not accounting for scr refresh
                    indicatorTitle.tStopRefresh = tThisFlipGlobal  # on global time
                    indicatorTitle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'indicatorTitle.stopped')
                    # update status
                    indicatorTitle.status = FINISHED
                    indicatorTitle.setAutoDraw(False)
            
            # *indicatorInfo* updates
            
            # if indicatorInfo is starting this frame...
            if indicatorInfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                indicatorInfo.frameNStart = frameN  # exact frame index
                indicatorInfo.tStart = t  # local t and not account for scr refresh
                indicatorInfo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(indicatorInfo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'indicatorInfo.started')
                # update status
                indicatorInfo.status = STARTED
                indicatorInfo.setAutoDraw(True)
            
            # if indicatorInfo is active this frame...
            if indicatorInfo.status == STARTED:
                # update params
                pass
            
            # if indicatorInfo is stopping this frame...
            if indicatorInfo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > indicatorInfo.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    indicatorInfo.tStop = t  # not accounting for scr refresh
                    indicatorInfo.tStopRefresh = tThisFlipGlobal  # on global time
                    indicatorInfo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'indicatorInfo.stopped')
                    # update status
                    indicatorInfo.status = FINISHED
                    indicatorInfo.setAutoDraw(False)
            
            # *indicatorImage* updates
            
            # if indicatorImage is starting this frame...
            if indicatorImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                indicatorImage.frameNStart = frameN  # exact frame index
                indicatorImage.tStart = t  # local t and not account for scr refresh
                indicatorImage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(indicatorImage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'indicatorImage.started')
                # update status
                indicatorImage.status = STARTED
                indicatorImage.setAutoDraw(True)
            
            # if indicatorImage is active this frame...
            if indicatorImage.status == STARTED:
                # update params
                pass
            
            # if indicatorImage is stopping this frame...
            if indicatorImage.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > indicatorImage.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    indicatorImage.tStop = t  # not accounting for scr refresh
                    indicatorImage.tStopRefresh = tThisFlipGlobal  # on global time
                    indicatorImage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'indicatorImage.stopped')
                    # update status
                    indicatorImage.status = FINISHED
                    indicatorImage.setAutoDraw(False)
            
            # *curvedLine_4* updates
            
            # if curvedLine_4 is starting this frame...
            if curvedLine_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_4.frameNStart = frameN  # exact frame index
                curvedLine_4.tStart = t  # local t and not account for scr refresh
                curvedLine_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_4.started')
                # update status
                curvedLine_4.status = STARTED
                curvedLine_4.setAutoDraw(True)
            
            # if curvedLine_4 is active this frame...
            if curvedLine_4.status == STARTED:
                # update params
                pass
            
            # if curvedLine_4 is stopping this frame...
            if curvedLine_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_4.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_4.tStop = t  # not accounting for scr refresh
                    curvedLine_4.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_4.stopped')
                    # update status
                    curvedLine_4.status = FINISHED
                    curvedLine_4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=indicators,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                indicators.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in indicators.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "indicators" ---
        for thisComponent in indicators.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for indicators
        indicators.tStop = globalClock.getTime(format='float')
        indicators.tStopRefresh = tThisFlipGlobal
        thisExp.addData('indicators.stopped', indicators.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if indicators.maxDurationReached:
            routineTimer.addTime(-indicators.maxDuration)
        elif indicators.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "pay_decision" ---
        # create an object to store info about Routine pay_decision
        pay_decision = data.Routine(
            name='pay_decision',
            components=[bgImage_4, balloonImage_3, moneyImage_3, moneyText_3, decisionText, payKey, curvedLine_5],
        )
        pay_decision.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for payKey
        payKey.keys = []
        payKey.rt = []
        _payKey_allKeys = []
        # store start times for pay_decision
        pay_decision.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pay_decision.tStart = globalClock.getTime(format='float')
        pay_decision.status = STARTED
        thisExp.addData('pay_decision.started', pay_decision.tStart)
        pay_decision.maxDuration = None
        # keep track of which components have finished
        pay_decisionComponents = pay_decision.components
        for thisComponent in pay_decision.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pay_decision" ---
        pay_decision.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_4* updates
            
            # if bgImage_4 is starting this frame...
            if bgImage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_4.frameNStart = frameN  # exact frame index
                bgImage_4.tStart = t  # local t and not account for scr refresh
                bgImage_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_4.started')
                # update status
                bgImage_4.status = STARTED
                bgImage_4.setAutoDraw(True)
            
            # if bgImage_4 is active this frame...
            if bgImage_4.status == STARTED:
                # update params
                pass
            
            # if bgImage_4 is stopping this frame...
            if bgImage_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_4.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_4.tStop = t  # not accounting for scr refresh
                    bgImage_4.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_4.stopped')
                    # update status
                    bgImage_4.status = FINISHED
                    bgImage_4.setAutoDraw(False)
            
            # *balloonImage_3* updates
            
            # if balloonImage_3 is starting this frame...
            if balloonImage_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloonImage_3.frameNStart = frameN  # exact frame index
                balloonImage_3.tStart = t  # local t and not account for scr refresh
                balloonImage_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloonImage_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloonImage_3.started')
                # update status
                balloonImage_3.status = STARTED
                balloonImage_3.setAutoDraw(True)
            
            # if balloonImage_3 is active this frame...
            if balloonImage_3.status == STARTED:
                # update params
                pass
            
            # if balloonImage_3 is stopping this frame...
            if balloonImage_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloonImage_3.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    balloonImage_3.tStop = t  # not accounting for scr refresh
                    balloonImage_3.tStopRefresh = tThisFlipGlobal  # on global time
                    balloonImage_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloonImage_3.stopped')
                    # update status
                    balloonImage_3.status = FINISHED
                    balloonImage_3.setAutoDraw(False)
            
            # *moneyImage_3* updates
            
            # if moneyImage_3 is starting this frame...
            if moneyImage_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_3.frameNStart = frameN  # exact frame index
                moneyImage_3.tStart = t  # local t and not account for scr refresh
                moneyImage_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_3.started')
                # update status
                moneyImage_3.status = STARTED
                moneyImage_3.setAutoDraw(True)
            
            # if moneyImage_3 is active this frame...
            if moneyImage_3.status == STARTED:
                # update params
                pass
            
            # if moneyImage_3 is stopping this frame...
            if moneyImage_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage_3.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage_3.tStop = t  # not accounting for scr refresh
                    moneyImage_3.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage_3.stopped')
                    # update status
                    moneyImage_3.status = FINISHED
                    moneyImage_3.setAutoDraw(False)
            
            # *moneyText_3* updates
            
            # if moneyText_3 is starting this frame...
            if moneyText_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_3.frameNStart = frameN  # exact frame index
                moneyText_3.tStart = t  # local t and not account for scr refresh
                moneyText_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_3.started')
                # update status
                moneyText_3.status = STARTED
                moneyText_3.setAutoDraw(True)
            
            # if moneyText_3 is active this frame...
            if moneyText_3.status == STARTED:
                # update params
                pass
            
            # if moneyText_3 is stopping this frame...
            if moneyText_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText_3.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText_3.tStop = t  # not accounting for scr refresh
                    moneyText_3.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText_3.stopped')
                    # update status
                    moneyText_3.status = FINISHED
                    moneyText_3.setAutoDraw(False)
            
            # *decisionText* updates
            
            # if decisionText is starting this frame...
            if decisionText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                decisionText.frameNStart = frameN  # exact frame index
                decisionText.tStart = t  # local t and not account for scr refresh
                decisionText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(decisionText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'decisionText.started')
                # update status
                decisionText.status = STARTED
                decisionText.setAutoDraw(True)
            
            # if decisionText is active this frame...
            if decisionText.status == STARTED:
                # update params
                pass
            
            # if decisionText is stopping this frame...
            if decisionText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > decisionText.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    decisionText.tStop = t  # not accounting for scr refresh
                    decisionText.tStopRefresh = tThisFlipGlobal  # on global time
                    decisionText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'decisionText.stopped')
                    # update status
                    decisionText.status = FINISHED
                    decisionText.setAutoDraw(False)
            
            # *payKey* updates
            waitOnFlip = False
            
            # if payKey is starting this frame...
            if payKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                payKey.frameNStart = frameN  # exact frame index
                payKey.tStart = t  # local t and not account for scr refresh
                payKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(payKey, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'payKey.started')
                # update status
                payKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(payKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(payKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if payKey is stopping this frame...
            if payKey.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > payKey.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    payKey.tStop = t  # not accounting for scr refresh
                    payKey.tStopRefresh = tThisFlipGlobal  # on global time
                    payKey.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'payKey.stopped')
                    # update status
                    payKey.status = FINISHED
                    payKey.status = FINISHED
            if payKey.status == STARTED and not waitOnFlip:
                theseKeys = payKey.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _payKey_allKeys.extend(theseKeys)
                if len(_payKey_allKeys):
                    payKey.keys = _payKey_allKeys[-1].name  # just the last key pressed
                    payKey.rt = _payKey_allKeys[-1].rt
                    payKey.duration = _payKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *curvedLine_5* updates
            
            # if curvedLine_5 is starting this frame...
            if curvedLine_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_5.frameNStart = frameN  # exact frame index
                curvedLine_5.tStart = t  # local t and not account for scr refresh
                curvedLine_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_5.started')
                # update status
                curvedLine_5.status = STARTED
                curvedLine_5.setAutoDraw(True)
            
            # if curvedLine_5 is active this frame...
            if curvedLine_5.status == STARTED:
                # update params
                pass
            
            # if curvedLine_5 is stopping this frame...
            if curvedLine_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_5.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_5.tStop = t  # not accounting for scr refresh
                    curvedLine_5.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_5.stopped')
                    # update status
                    curvedLine_5.status = FINISHED
                    curvedLine_5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pay_decision,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pay_decision.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pay_decision.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pay_decision" ---
        for thisComponent in pay_decision.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pay_decision
        pay_decision.tStop = globalClock.getTime(format='float')
        pay_decision.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pay_decision.stopped', pay_decision.tStop)
        # Run 'End Routine' code from payCode
        if payKey.keys == 'y':
            pay = 1
        else:
            pay = 0
        
        thisExp.addData('pay', pay)
        # check responses
        if payKey.keys in ['', [], None]:  # No response was made
            payKey.keys = None
        trialLoop.addData('payKey.keys',payKey.keys)
        if payKey.keys != None:  # we had a response
            trialLoop.addData('payKey.rt', payKey.rt)
            trialLoop.addData('payKey.duration', payKey.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pay_decision.maxDurationReached:
            routineTimer.addTime(-pay_decision.maxDuration)
        elif pay_decision.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "shield_phase" ---
        # create an object to store info about Routine shield_phase
        shield_phase = data.Routine(
            name='shield_phase',
            components=[bgImage_5, balloonImage_4, moneyImage_4, moneyText_4, shieldInfo, shieldCondition, shieldImage, curvedLine_6],
        )
        shield_phase.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        shieldInfo.setText(("Paid: glitter shown" if payKey.keys == "y" else "Not paid: shield only"))
        shieldCondition.setText(("" if condition is None else str(condition)) + " | " + ("" if airship_type is None else str(airship_type)))
        # store start times for shield_phase
        shield_phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        shield_phase.tStart = globalClock.getTime(format='float')
        shield_phase.status = STARTED
        thisExp.addData('shield_phase.started', shield_phase.tStart)
        shield_phase.maxDuration = None
        # keep track of which components have finished
        shield_phaseComponents = shield_phase.components
        for thisComponent in shield_phase.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "shield_phase" ---
        shield_phase.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_5* updates
            
            # if bgImage_5 is starting this frame...
            if bgImage_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_5.frameNStart = frameN  # exact frame index
                bgImage_5.tStart = t  # local t and not account for scr refresh
                bgImage_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_5.started')
                # update status
                bgImage_5.status = STARTED
                bgImage_5.setAutoDraw(True)
            
            # if bgImage_5 is active this frame...
            if bgImage_5.status == STARTED:
                # update params
                pass
            
            # if bgImage_5 is stopping this frame...
            if bgImage_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_5.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_5.tStop = t  # not accounting for scr refresh
                    bgImage_5.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_5.stopped')
                    # update status
                    bgImage_5.status = FINISHED
                    bgImage_5.setAutoDraw(False)
            
            # *balloonImage_4* updates
            
            # if balloonImage_4 is starting this frame...
            if balloonImage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloonImage_4.frameNStart = frameN  # exact frame index
                balloonImage_4.tStart = t  # local t and not account for scr refresh
                balloonImage_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloonImage_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloonImage_4.started')
                # update status
                balloonImage_4.status = STARTED
                balloonImage_4.setAutoDraw(True)
            
            # if balloonImage_4 is active this frame...
            if balloonImage_4.status == STARTED:
                # update params
                pass
            
            # if balloonImage_4 is stopping this frame...
            if balloonImage_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloonImage_4.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    balloonImage_4.tStop = t  # not accounting for scr refresh
                    balloonImage_4.tStopRefresh = tThisFlipGlobal  # on global time
                    balloonImage_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloonImage_4.stopped')
                    # update status
                    balloonImage_4.status = FINISHED
                    balloonImage_4.setAutoDraw(False)
            
            # *moneyImage_4* updates
            
            # if moneyImage_4 is starting this frame...
            if moneyImage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_4.frameNStart = frameN  # exact frame index
                moneyImage_4.tStart = t  # local t and not account for scr refresh
                moneyImage_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_4.started')
                # update status
                moneyImage_4.status = STARTED
                moneyImage_4.setAutoDraw(True)
            
            # if moneyImage_4 is active this frame...
            if moneyImage_4.status == STARTED:
                # update params
                pass
            
            # if moneyImage_4 is stopping this frame...
            if moneyImage_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage_4.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage_4.tStop = t  # not accounting for scr refresh
                    moneyImage_4.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage_4.stopped')
                    # update status
                    moneyImage_4.status = FINISHED
                    moneyImage_4.setAutoDraw(False)
            
            # *moneyText_4* updates
            
            # if moneyText_4 is starting this frame...
            if moneyText_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_4.frameNStart = frameN  # exact frame index
                moneyText_4.tStart = t  # local t and not account for scr refresh
                moneyText_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_4.started')
                # update status
                moneyText_4.status = STARTED
                moneyText_4.setAutoDraw(True)
            
            # if moneyText_4 is active this frame...
            if moneyText_4.status == STARTED:
                # update params
                pass
            
            # if moneyText_4 is stopping this frame...
            if moneyText_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText_4.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText_4.tStop = t  # not accounting for scr refresh
                    moneyText_4.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText_4.stopped')
                    # update status
                    moneyText_4.status = FINISHED
                    moneyText_4.setAutoDraw(False)
            
            # *shieldInfo* updates
            
            # if shieldInfo is starting this frame...
            if shieldInfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                shieldInfo.frameNStart = frameN  # exact frame index
                shieldInfo.tStart = t  # local t and not account for scr refresh
                shieldInfo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shieldInfo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shieldInfo.started')
                # update status
                shieldInfo.status = STARTED
                shieldInfo.setAutoDraw(True)
            
            # if shieldInfo is active this frame...
            if shieldInfo.status == STARTED:
                # update params
                pass
            
            # if shieldInfo is stopping this frame...
            if shieldInfo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shieldInfo.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    shieldInfo.tStop = t  # not accounting for scr refresh
                    shieldInfo.tStopRefresh = tThisFlipGlobal  # on global time
                    shieldInfo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shieldInfo.stopped')
                    # update status
                    shieldInfo.status = FINISHED
                    shieldInfo.setAutoDraw(False)
            
            # *shieldCondition* updates
            
            # if shieldCondition is starting this frame...
            if shieldCondition.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                shieldCondition.frameNStart = frameN  # exact frame index
                shieldCondition.tStart = t  # local t and not account for scr refresh
                shieldCondition.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shieldCondition, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shieldCondition.started')
                # update status
                shieldCondition.status = STARTED
                shieldCondition.setAutoDraw(True)
            
            # if shieldCondition is active this frame...
            if shieldCondition.status == STARTED:
                # update params
                pass
            
            # if shieldCondition is stopping this frame...
            if shieldCondition.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shieldCondition.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    shieldCondition.tStop = t  # not accounting for scr refresh
                    shieldCondition.tStopRefresh = tThisFlipGlobal  # on global time
                    shieldCondition.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shieldCondition.stopped')
                    # update status
                    shieldCondition.status = FINISHED
                    shieldCondition.setAutoDraw(False)
            
            # *shieldImage* updates
            
            # if shieldImage is starting this frame...
            if shieldImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                shieldImage.frameNStart = frameN  # exact frame index
                shieldImage.tStart = t  # local t and not account for scr refresh
                shieldImage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shieldImage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shieldImage.started')
                # update status
                shieldImage.status = STARTED
                shieldImage.setAutoDraw(True)
            
            # if shieldImage is active this frame...
            if shieldImage.status == STARTED:
                # update params
                pass
            
            # if shieldImage is stopping this frame...
            if shieldImage.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shieldImage.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    shieldImage.tStop = t  # not accounting for scr refresh
                    shieldImage.tStopRefresh = tThisFlipGlobal  # on global time
                    shieldImage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shieldImage.stopped')
                    # update status
                    shieldImage.status = FINISHED
                    shieldImage.setAutoDraw(False)
            
            # *curvedLine_6* updates
            
            # if curvedLine_6 is starting this frame...
            if curvedLine_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_6.frameNStart = frameN  # exact frame index
                curvedLine_6.tStart = t  # local t and not account for scr refresh
                curvedLine_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_6.started')
                # update status
                curvedLine_6.status = STARTED
                curvedLine_6.setAutoDraw(True)
            
            # if curvedLine_6 is active this frame...
            if curvedLine_6.status == STARTED:
                # update params
                pass
            
            # if curvedLine_6 is stopping this frame...
            if curvedLine_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_6.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_6.tStop = t  # not accounting for scr refresh
                    curvedLine_6.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_6.stopped')
                    # update status
                    curvedLine_6.status = FINISHED
                    curvedLine_6.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=shield_phase,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                shield_phase.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in shield_phase.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "shield_phase" ---
        for thisComponent in shield_phase.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for shield_phase
        shield_phase.tStop = globalClock.getTime(format='float')
        shield_phase.tStopRefresh = tThisFlipGlobal
        thisExp.addData('shield_phase.stopped', shield_phase.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if shield_phase.maxDurationReached:
            routineTimer.addTime(-shield_phase.maxDuration)
        elif shield_phase.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "airship_arrival" ---
        # create an object to store info about Routine airship_arrival
        airship_arrival = data.Routine(
            name='airship_arrival',
            components=[bgImage_6, balloonImage_5, moneyImage_5, moneyText_5, arrivalText, arrivalInfo, partialText, noneText, curvedLine_7, airshipImage],
        )
        airship_arrival.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from airshipCode
        import math
        
        safeOutcome = 0 if outcome is None else float(outcome)
        safeOutcome = max(0, min(120, safeOutcome))
        
        # remap 0–120 to 0–180 degrees on a semicircle
        angle_deg = (safeOutcome / 120.0) * 180.0
        theta = math.radians(angle_deg)
        
        center_x = 0.0
        center_y = -0.35
        radius = 0.55
        
        target_x = center_x - radius * math.cos(theta)
        target_y = center_y + radius * math.sin(theta)
        
        # start above final position
        start_x = target_x
        start_y = 0.72
        
        airshipImage.setImage(airshipFile)
        airshipImage.setPos((start_x, start_y))
        airshipImage.interpolate = False
        
        arrivalText.setOpacity(1 if condition == "Full" else 0)
        arrivalInfo.setOpacity(1 if condition == "Full" else 0)
        arrivalInfo.setText("Attack angle: " + str(round(safeOutcome, 1)))
        partialText.setOpacity(1 if condition == "Partial" else 0)
        noneText.setOpacity(1 if condition == "None" else 0)
        arrivalText.setOpacity((1 if condition == "Full" else 0))
        arrivalInfo.setOpacity(1 if condition == "Full" else 0)
        arrivalInfo.setText(("Attack angle: " + str(round(outcome, 1))))
        partialText.setOpacity(1 if condition == "Partial" else 0)
        noneText.setOpacity(1 if condition == "None" else 0)
        airshipImage.setImage(airshipFile)
        # store start times for airship_arrival
        airship_arrival.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        airship_arrival.tStart = globalClock.getTime(format='float')
        airship_arrival.status = STARTED
        thisExp.addData('airship_arrival.started', airship_arrival.tStart)
        airship_arrival.maxDuration = None
        # keep track of which components have finished
        airship_arrivalComponents = airship_arrival.components
        for thisComponent in airship_arrival.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "airship_arrival" ---
        airship_arrival.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from airshipCode
            progress = min(t / 2.2, 1.0)
            
            current_x = start_x + progress * (target_x - start_x)
            current_y = start_y + progress * (target_y - start_y)
            
            airshipImage.setPos((current_x, current_y))
            
            # *bgImage_6* updates
            
            # if bgImage_6 is starting this frame...
            if bgImage_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_6.frameNStart = frameN  # exact frame index
                bgImage_6.tStart = t  # local t and not account for scr refresh
                bgImage_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_6.started')
                # update status
                bgImage_6.status = STARTED
                bgImage_6.setAutoDraw(True)
            
            # if bgImage_6 is active this frame...
            if bgImage_6.status == STARTED:
                # update params
                pass
            
            # if bgImage_6 is stopping this frame...
            if bgImage_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_6.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_6.tStop = t  # not accounting for scr refresh
                    bgImage_6.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_6.stopped')
                    # update status
                    bgImage_6.status = FINISHED
                    bgImage_6.setAutoDraw(False)
            
            # *balloonImage_5* updates
            
            # if balloonImage_5 is starting this frame...
            if balloonImage_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloonImage_5.frameNStart = frameN  # exact frame index
                balloonImage_5.tStart = t  # local t and not account for scr refresh
                balloonImage_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloonImage_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloonImage_5.started')
                # update status
                balloonImage_5.status = STARTED
                balloonImage_5.setAutoDraw(True)
            
            # if balloonImage_5 is active this frame...
            if balloonImage_5.status == STARTED:
                # update params
                pass
            
            # if balloonImage_5 is stopping this frame...
            if balloonImage_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloonImage_5.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    balloonImage_5.tStop = t  # not accounting for scr refresh
                    balloonImage_5.tStopRefresh = tThisFlipGlobal  # on global time
                    balloonImage_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloonImage_5.stopped')
                    # update status
                    balloonImage_5.status = FINISHED
                    balloonImage_5.setAutoDraw(False)
            
            # *moneyImage_5* updates
            
            # if moneyImage_5 is starting this frame...
            if moneyImage_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_5.frameNStart = frameN  # exact frame index
                moneyImage_5.tStart = t  # local t and not account for scr refresh
                moneyImage_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_5.started')
                # update status
                moneyImage_5.status = STARTED
                moneyImage_5.setAutoDraw(True)
            
            # if moneyImage_5 is active this frame...
            if moneyImage_5.status == STARTED:
                # update params
                pass
            
            # if moneyImage_5 is stopping this frame...
            if moneyImage_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage_5.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage_5.tStop = t  # not accounting for scr refresh
                    moneyImage_5.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage_5.stopped')
                    # update status
                    moneyImage_5.status = FINISHED
                    moneyImage_5.setAutoDraw(False)
            
            # *moneyText_5* updates
            
            # if moneyText_5 is starting this frame...
            if moneyText_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_5.frameNStart = frameN  # exact frame index
                moneyText_5.tStart = t  # local t and not account for scr refresh
                moneyText_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_5.started')
                # update status
                moneyText_5.status = STARTED
                moneyText_5.setAutoDraw(True)
            
            # if moneyText_5 is active this frame...
            if moneyText_5.status == STARTED:
                # update params
                pass
            
            # if moneyText_5 is stopping this frame...
            if moneyText_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText_5.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText_5.tStop = t  # not accounting for scr refresh
                    moneyText_5.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText_5.stopped')
                    # update status
                    moneyText_5.status = FINISHED
                    moneyText_5.setAutoDraw(False)
            
            # *arrivalText* updates
            
            # if arrivalText is starting this frame...
            if arrivalText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrivalText.frameNStart = frameN  # exact frame index
                arrivalText.tStart = t  # local t and not account for scr refresh
                arrivalText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrivalText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrivalText.started')
                # update status
                arrivalText.status = STARTED
                arrivalText.setAutoDraw(True)
            
            # if arrivalText is active this frame...
            if arrivalText.status == STARTED:
                # update params
                pass
            
            # if arrivalText is stopping this frame...
            if arrivalText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrivalText.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    arrivalText.tStop = t  # not accounting for scr refresh
                    arrivalText.tStopRefresh = tThisFlipGlobal  # on global time
                    arrivalText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrivalText.stopped')
                    # update status
                    arrivalText.status = FINISHED
                    arrivalText.setAutoDraw(False)
            
            # *arrivalInfo* updates
            
            # if arrivalInfo is starting this frame...
            if arrivalInfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrivalInfo.frameNStart = frameN  # exact frame index
                arrivalInfo.tStart = t  # local t and not account for scr refresh
                arrivalInfo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrivalInfo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrivalInfo.started')
                # update status
                arrivalInfo.status = STARTED
                arrivalInfo.setAutoDraw(True)
            
            # if arrivalInfo is active this frame...
            if arrivalInfo.status == STARTED:
                # update params
                pass
            
            # if arrivalInfo is stopping this frame...
            if arrivalInfo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrivalInfo.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    arrivalInfo.tStop = t  # not accounting for scr refresh
                    arrivalInfo.tStopRefresh = tThisFlipGlobal  # on global time
                    arrivalInfo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrivalInfo.stopped')
                    # update status
                    arrivalInfo.status = FINISHED
                    arrivalInfo.setAutoDraw(False)
            
            # *partialText* updates
            
            # if partialText is starting this frame...
            if partialText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                partialText.frameNStart = frameN  # exact frame index
                partialText.tStart = t  # local t and not account for scr refresh
                partialText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(partialText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'partialText.started')
                # update status
                partialText.status = STARTED
                partialText.setAutoDraw(True)
            
            # if partialText is active this frame...
            if partialText.status == STARTED:
                # update params
                pass
            
            # if partialText is stopping this frame...
            if partialText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > partialText.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    partialText.tStop = t  # not accounting for scr refresh
                    partialText.tStopRefresh = tThisFlipGlobal  # on global time
                    partialText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'partialText.stopped')
                    # update status
                    partialText.status = FINISHED
                    partialText.setAutoDraw(False)
            
            # *noneText* updates
            
            # if noneText is starting this frame...
            if noneText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noneText.frameNStart = frameN  # exact frame index
                noneText.tStart = t  # local t and not account for scr refresh
                noneText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noneText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'noneText.started')
                # update status
                noneText.status = STARTED
                noneText.setAutoDraw(True)
            
            # if noneText is active this frame...
            if noneText.status == STARTED:
                # update params
                pass
            
            # if noneText is stopping this frame...
            if noneText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noneText.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    noneText.tStop = t  # not accounting for scr refresh
                    noneText.tStopRefresh = tThisFlipGlobal  # on global time
                    noneText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'noneText.stopped')
                    # update status
                    noneText.status = FINISHED
                    noneText.setAutoDraw(False)
            
            # *curvedLine_7* updates
            
            # if curvedLine_7 is starting this frame...
            if curvedLine_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_7.frameNStart = frameN  # exact frame index
                curvedLine_7.tStart = t  # local t and not account for scr refresh
                curvedLine_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_7.started')
                # update status
                curvedLine_7.status = STARTED
                curvedLine_7.setAutoDraw(True)
            
            # if curvedLine_7 is active this frame...
            if curvedLine_7.status == STARTED:
                # update params
                pass
            
            # if curvedLine_7 is stopping this frame...
            if curvedLine_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_7.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_7.tStop = t  # not accounting for scr refresh
                    curvedLine_7.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_7.stopped')
                    # update status
                    curvedLine_7.status = FINISHED
                    curvedLine_7.setAutoDraw(False)
            
            # *airshipImage* updates
            
            # if airshipImage is starting this frame...
            if airshipImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                airshipImage.frameNStart = frameN  # exact frame index
                airshipImage.tStart = t  # local t and not account for scr refresh
                airshipImage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(airshipImage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'airshipImage.started')
                # update status
                airshipImage.status = STARTED
                airshipImage.setAutoDraw(True)
            
            # if airshipImage is active this frame...
            if airshipImage.status == STARTED:
                # update params
                pass
            
            # if airshipImage is stopping this frame...
            if airshipImage.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > airshipImage.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    airshipImage.tStop = t  # not accounting for scr refresh
                    airshipImage.tStopRefresh = tThisFlipGlobal  # on global time
                    airshipImage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'airshipImage.stopped')
                    # update status
                    airshipImage.status = FINISHED
                    airshipImage.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=airship_arrival,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                airship_arrival.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in airship_arrival.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "airship_arrival" ---
        for thisComponent in airship_arrival.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for airship_arrival
        airship_arrival.tStop = globalClock.getTime(format='float')
        airship_arrival.tStopRefresh = tThisFlipGlobal
        thisExp.addData('airship_arrival.stopped', airship_arrival.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if airship_arrival.maxDurationReached:
            routineTimer.addTime(-airship_arrival.maxDuration)
        elif airship_arrival.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "airship_attack" ---
        # create an object to store info about Routine airship_attack
        airship_attack = data.Routine(
            name='airship_attack',
            components=[bgImage_7, balloonImage_6, moneyImage_6, moneyText_6, attackText, attackInfo, partialAttackText, noneAttackText, curvedLine_8, airshipImage_2],
        )
        airship_attack.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from airshipCode_2
        import math
        
        safeOutcome = 0 if outcome is None else float(outcome)
        safeOutcome = max(0, min(120, safeOutcome))
        
        # remap 0–120 to 0–180 degrees on a semicircle
        angle_deg = (safeOutcome / 120.0) * 180.0
        theta = math.radians(angle_deg)
        
        center_x = 0.0
        center_y = -0.35
        radius = 0.55
        
        target_x = center_x - radius * math.cos(theta)
        target_y = center_y + radius * math.sin(theta)
        
        airshipImage_2.setImage(airshipFile)
        airshipImage_2.setPos((target_x, target_y))
        airshipImage_2.interpolate = False
        
        attackText.setOpacity(1 if condition == "Partial" else 0)
        attackInfo.setOpacity(1 if condition == "Full" else 0)
        attackInfo.setText(
            ("" if condition is None else str(condition))
            + " | "
            + ("" if airship_type is None else str(airship_type))
            + " | angle="
            + str(round(safeOutcome, 1))
        )
        partialAttackText.setOpacity(1 if condition == "Partial" else 0)
        noneAttackText.setOpacity(1 if condition == "None" else 0)
        attackText.setOpacity(1 if condition == "Partial" else 0)
        attackInfo.setOpacity(1 if condition == "Full" else 0)
        attackInfo.setText(("" if condition is None else str(condition)) + " | " + ("" if airship_type is None else str(airship_type)) + " | angle=" + str(round(outcome, 1)))
        partialAttackText.setOpacity(1 if condition == "Partial" else 0)
        noneAttackText.setOpacity(1 if condition == "None" else 0)
        airshipImage_2.setImage(airshipFile)
        # store start times for airship_attack
        airship_attack.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        airship_attack.tStart = globalClock.getTime(format='float')
        airship_attack.status = STARTED
        thisExp.addData('airship_attack.started', airship_attack.tStart)
        airship_attack.maxDuration = None
        # keep track of which components have finished
        airship_attackComponents = airship_attack.components
        for thisComponent in airship_attack.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "airship_attack" ---
        airship_attack.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_7* updates
            
            # if bgImage_7 is starting this frame...
            if bgImage_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_7.frameNStart = frameN  # exact frame index
                bgImage_7.tStart = t  # local t and not account for scr refresh
                bgImage_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_7.started')
                # update status
                bgImage_7.status = STARTED
                bgImage_7.setAutoDraw(True)
            
            # if bgImage_7 is active this frame...
            if bgImage_7.status == STARTED:
                # update params
                pass
            
            # if bgImage_7 is stopping this frame...
            if bgImage_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_7.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_7.tStop = t  # not accounting for scr refresh
                    bgImage_7.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_7.stopped')
                    # update status
                    bgImage_7.status = FINISHED
                    bgImage_7.setAutoDraw(False)
            
            # *balloonImage_6* updates
            
            # if balloonImage_6 is starting this frame...
            if balloonImage_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloonImage_6.frameNStart = frameN  # exact frame index
                balloonImage_6.tStart = t  # local t and not account for scr refresh
                balloonImage_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloonImage_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloonImage_6.started')
                # update status
                balloonImage_6.status = STARTED
                balloonImage_6.setAutoDraw(True)
            
            # if balloonImage_6 is active this frame...
            if balloonImage_6.status == STARTED:
                # update params
                pass
            
            # if balloonImage_6 is stopping this frame...
            if balloonImage_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloonImage_6.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    balloonImage_6.tStop = t  # not accounting for scr refresh
                    balloonImage_6.tStopRefresh = tThisFlipGlobal  # on global time
                    balloonImage_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloonImage_6.stopped')
                    # update status
                    balloonImage_6.status = FINISHED
                    balloonImage_6.setAutoDraw(False)
            
            # *moneyImage_6* updates
            
            # if moneyImage_6 is starting this frame...
            if moneyImage_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_6.frameNStart = frameN  # exact frame index
                moneyImage_6.tStart = t  # local t and not account for scr refresh
                moneyImage_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_6.started')
                # update status
                moneyImage_6.status = STARTED
                moneyImage_6.setAutoDraw(True)
            
            # if moneyImage_6 is active this frame...
            if moneyImage_6.status == STARTED:
                # update params
                pass
            
            # if moneyImage_6 is stopping this frame...
            if moneyImage_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage_6.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage_6.tStop = t  # not accounting for scr refresh
                    moneyImage_6.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage_6.stopped')
                    # update status
                    moneyImage_6.status = FINISHED
                    moneyImage_6.setAutoDraw(False)
            
            # *moneyText_6* updates
            
            # if moneyText_6 is starting this frame...
            if moneyText_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_6.frameNStart = frameN  # exact frame index
                moneyText_6.tStart = t  # local t and not account for scr refresh
                moneyText_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_6.started')
                # update status
                moneyText_6.status = STARTED
                moneyText_6.setAutoDraw(True)
            
            # if moneyText_6 is active this frame...
            if moneyText_6.status == STARTED:
                # update params
                pass
            
            # if moneyText_6 is stopping this frame...
            if moneyText_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText_6.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText_6.tStop = t  # not accounting for scr refresh
                    moneyText_6.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText_6.stopped')
                    # update status
                    moneyText_6.status = FINISHED
                    moneyText_6.setAutoDraw(False)
            
            # *attackText* updates
            
            # if attackText is starting this frame...
            if attackText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                attackText.frameNStart = frameN  # exact frame index
                attackText.tStart = t  # local t and not account for scr refresh
                attackText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(attackText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'attackText.started')
                # update status
                attackText.status = STARTED
                attackText.setAutoDraw(True)
            
            # if attackText is active this frame...
            if attackText.status == STARTED:
                # update params
                pass
            
            # if attackText is stopping this frame...
            if attackText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > attackText.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    attackText.tStop = t  # not accounting for scr refresh
                    attackText.tStopRefresh = tThisFlipGlobal  # on global time
                    attackText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'attackText.stopped')
                    # update status
                    attackText.status = FINISHED
                    attackText.setAutoDraw(False)
            
            # *attackInfo* updates
            
            # if attackInfo is starting this frame...
            if attackInfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                attackInfo.frameNStart = frameN  # exact frame index
                attackInfo.tStart = t  # local t and not account for scr refresh
                attackInfo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(attackInfo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'attackInfo.started')
                # update status
                attackInfo.status = STARTED
                attackInfo.setAutoDraw(True)
            
            # if attackInfo is active this frame...
            if attackInfo.status == STARTED:
                # update params
                pass
            
            # if attackInfo is stopping this frame...
            if attackInfo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > attackInfo.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    attackInfo.tStop = t  # not accounting for scr refresh
                    attackInfo.tStopRefresh = tThisFlipGlobal  # on global time
                    attackInfo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'attackInfo.stopped')
                    # update status
                    attackInfo.status = FINISHED
                    attackInfo.setAutoDraw(False)
            
            # *partialAttackText* updates
            
            # if partialAttackText is starting this frame...
            if partialAttackText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                partialAttackText.frameNStart = frameN  # exact frame index
                partialAttackText.tStart = t  # local t and not account for scr refresh
                partialAttackText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(partialAttackText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'partialAttackText.started')
                # update status
                partialAttackText.status = STARTED
                partialAttackText.setAutoDraw(True)
            
            # if partialAttackText is active this frame...
            if partialAttackText.status == STARTED:
                # update params
                pass
            
            # if partialAttackText is stopping this frame...
            if partialAttackText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > partialAttackText.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    partialAttackText.tStop = t  # not accounting for scr refresh
                    partialAttackText.tStopRefresh = tThisFlipGlobal  # on global time
                    partialAttackText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'partialAttackText.stopped')
                    # update status
                    partialAttackText.status = FINISHED
                    partialAttackText.setAutoDraw(False)
            
            # *noneAttackText* updates
            
            # if noneAttackText is starting this frame...
            if noneAttackText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noneAttackText.frameNStart = frameN  # exact frame index
                noneAttackText.tStart = t  # local t and not account for scr refresh
                noneAttackText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noneAttackText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'noneAttackText.started')
                # update status
                noneAttackText.status = STARTED
                noneAttackText.setAutoDraw(True)
            
            # if noneAttackText is active this frame...
            if noneAttackText.status == STARTED:
                # update params
                pass
            
            # if noneAttackText is stopping this frame...
            if noneAttackText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noneAttackText.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    noneAttackText.tStop = t  # not accounting for scr refresh
                    noneAttackText.tStopRefresh = tThisFlipGlobal  # on global time
                    noneAttackText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'noneAttackText.stopped')
                    # update status
                    noneAttackText.status = FINISHED
                    noneAttackText.setAutoDraw(False)
            
            # *curvedLine_8* updates
            
            # if curvedLine_8 is starting this frame...
            if curvedLine_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_8.frameNStart = frameN  # exact frame index
                curvedLine_8.tStart = t  # local t and not account for scr refresh
                curvedLine_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_8.started')
                # update status
                curvedLine_8.status = STARTED
                curvedLine_8.setAutoDraw(True)
            
            # if curvedLine_8 is active this frame...
            if curvedLine_8.status == STARTED:
                # update params
                pass
            
            # if curvedLine_8 is stopping this frame...
            if curvedLine_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_8.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_8.tStop = t  # not accounting for scr refresh
                    curvedLine_8.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_8.stopped')
                    # update status
                    curvedLine_8.status = FINISHED
                    curvedLine_8.setAutoDraw(False)
            
            # *airshipImage_2* updates
            
            # if airshipImage_2 is starting this frame...
            if airshipImage_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                airshipImage_2.frameNStart = frameN  # exact frame index
                airshipImage_2.tStart = t  # local t and not account for scr refresh
                airshipImage_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(airshipImage_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'airshipImage_2.started')
                # update status
                airshipImage_2.status = STARTED
                airshipImage_2.setAutoDraw(True)
            
            # if airshipImage_2 is active this frame...
            if airshipImage_2.status == STARTED:
                # update params
                pass
            
            # if airshipImage_2 is stopping this frame...
            if airshipImage_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > airshipImage_2.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    airshipImage_2.tStop = t  # not accounting for scr refresh
                    airshipImage_2.tStopRefresh = tThisFlipGlobal  # on global time
                    airshipImage_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'airshipImage_2.stopped')
                    # update status
                    airshipImage_2.status = FINISHED
                    airshipImage_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=airship_attack,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                airship_attack.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in airship_attack.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "airship_attack" ---
        for thisComponent in airship_attack.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for airship_attack
        airship_attack.tStop = globalClock.getTime(format='float')
        airship_attack.tStopRefresh = tThisFlipGlobal
        thisExp.addData('airship_attack.stopped', airship_attack.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if airship_attack.maxDurationReached:
            routineTimer.addTime(-airship_attack.maxDuration)
        elif airship_attack.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "trial_end" ---
        # create an object to store info about Routine trial_end
        trial_end = data.Routine(
            name='trial_end',
            components=[bgImage_8, moneyImage_7, moneyText_7, trialEndText, curvedLine_9],
        )
        trial_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for trial_end
        trial_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_end.tStart = globalClock.getTime(format='float')
        trial_end.status = STARTED
        thisExp.addData('trial_end.started', trial_end.tStart)
        trial_end.maxDuration = None
        # keep track of which components have finished
        trial_endComponents = trial_end.components
        for thisComponent in trial_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_end" ---
        trial_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_8* updates
            
            # if bgImage_8 is starting this frame...
            if bgImage_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_8.frameNStart = frameN  # exact frame index
                bgImage_8.tStart = t  # local t and not account for scr refresh
                bgImage_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_8.started')
                # update status
                bgImage_8.status = STARTED
                bgImage_8.setAutoDraw(True)
            
            # if bgImage_8 is active this frame...
            if bgImage_8.status == STARTED:
                # update params
                pass
            
            # if bgImage_8 is stopping this frame...
            if bgImage_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgImage_8.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    bgImage_8.tStop = t  # not accounting for scr refresh
                    bgImage_8.tStopRefresh = tThisFlipGlobal  # on global time
                    bgImage_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgImage_8.stopped')
                    # update status
                    bgImage_8.status = FINISHED
                    bgImage_8.setAutoDraw(False)
            
            # *moneyImage_7* updates
            
            # if moneyImage_7 is starting this frame...
            if moneyImage_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_7.frameNStart = frameN  # exact frame index
                moneyImage_7.tStart = t  # local t and not account for scr refresh
                moneyImage_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_7.started')
                # update status
                moneyImage_7.status = STARTED
                moneyImage_7.setAutoDraw(True)
            
            # if moneyImage_7 is active this frame...
            if moneyImage_7.status == STARTED:
                # update params
                pass
            
            # if moneyImage_7 is stopping this frame...
            if moneyImage_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyImage_7.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyImage_7.tStop = t  # not accounting for scr refresh
                    moneyImage_7.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyImage_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyImage_7.stopped')
                    # update status
                    moneyImage_7.status = FINISHED
                    moneyImage_7.setAutoDraw(False)
            
            # *moneyText_7* updates
            
            # if moneyText_7 is starting this frame...
            if moneyText_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_7.frameNStart = frameN  # exact frame index
                moneyText_7.tStart = t  # local t and not account for scr refresh
                moneyText_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_7.started')
                # update status
                moneyText_7.status = STARTED
                moneyText_7.setAutoDraw(True)
            
            # if moneyText_7 is active this frame...
            if moneyText_7.status == STARTED:
                # update params
                pass
            
            # if moneyText_7 is stopping this frame...
            if moneyText_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > moneyText_7.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    moneyText_7.tStop = t  # not accounting for scr refresh
                    moneyText_7.tStopRefresh = tThisFlipGlobal  # on global time
                    moneyText_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'moneyText_7.stopped')
                    # update status
                    moneyText_7.status = FINISHED
                    moneyText_7.setAutoDraw(False)
            
            # *trialEndText* updates
            
            # if trialEndText is starting this frame...
            if trialEndText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trialEndText.frameNStart = frameN  # exact frame index
                trialEndText.tStart = t  # local t and not account for scr refresh
                trialEndText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trialEndText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trialEndText.started')
                # update status
                trialEndText.status = STARTED
                trialEndText.setAutoDraw(True)
            
            # if trialEndText is active this frame...
            if trialEndText.status == STARTED:
                # update params
                pass
            
            # if trialEndText is stopping this frame...
            if trialEndText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trialEndText.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    trialEndText.tStop = t  # not accounting for scr refresh
                    trialEndText.tStopRefresh = tThisFlipGlobal  # on global time
                    trialEndText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trialEndText.stopped')
                    # update status
                    trialEndText.status = FINISHED
                    trialEndText.setAutoDraw(False)
            
            # *curvedLine_9* updates
            
            # if curvedLine_9 is starting this frame...
            if curvedLine_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_9.frameNStart = frameN  # exact frame index
                curvedLine_9.tStart = t  # local t and not account for scr refresh
                curvedLine_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_9.started')
                # update status
                curvedLine_9.status = STARTED
                curvedLine_9.setAutoDraw(True)
            
            # if curvedLine_9 is active this frame...
            if curvedLine_9.status == STARTED:
                # update params
                pass
            
            # if curvedLine_9 is stopping this frame...
            if curvedLine_9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > curvedLine_9.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    curvedLine_9.tStop = t  # not accounting for scr refresh
                    curvedLine_9.tStopRefresh = tThisFlipGlobal  # on global time
                    curvedLine_9.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'curvedLine_9.stopped')
                    # update status
                    curvedLine_9.status = FINISHED
                    curvedLine_9.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_end,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_end" ---
        for thisComponent in trial_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_end
        trial_end.tStop = globalClock.getTime(format='float')
        trial_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_end.stopped', trial_end.tStop)
        # Run 'End Routine' code from code
        showBreak = trial_number in [40, 80]
        # Run 'End Routine' code from saveData
        thisExp.addData('condition', condition)
        thisExp.addData('pay', pay if 'pay' in locals() else '')
        thisExp.addData('outcome', outcome)
        thisExp.addData('airship_type', airship_type)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial_end.maxDurationReached:
            routineTimer.addTime(-trial_end.maxDuration)
        elif trial_end.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "break_screen" ---
        # create an object to store info about Routine break_screen
        break_screen = data.Routine(
            name='break_screen',
            components=[bgImage_9, moneyImage_8, moneyText_8, breakText, breakKey, curvedLine_10],
        )
        break_screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        if not showBreak:
            continueRoutine = False
        # create starting attributes for breakKey
        breakKey.keys = []
        breakKey.rt = []
        _breakKey_allKeys = []
        # store start times for break_screen
        break_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        break_screen.tStart = globalClock.getTime(format='float')
        break_screen.status = STARTED
        thisExp.addData('break_screen.started', break_screen.tStart)
        break_screen.maxDuration = None
        # keep track of which components have finished
        break_screenComponents = break_screen.components
        for thisComponent in break_screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_screen" ---
        break_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgImage_9* updates
            
            # if bgImage_9 is starting this frame...
            if bgImage_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgImage_9.frameNStart = frameN  # exact frame index
                bgImage_9.tStart = t  # local t and not account for scr refresh
                bgImage_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgImage_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgImage_9.started')
                # update status
                bgImage_9.status = STARTED
                bgImage_9.setAutoDraw(True)
            
            # if bgImage_9 is active this frame...
            if bgImage_9.status == STARTED:
                # update params
                pass
            
            # *moneyImage_8* updates
            
            # if moneyImage_8 is starting this frame...
            if moneyImage_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyImage_8.frameNStart = frameN  # exact frame index
                moneyImage_8.tStart = t  # local t and not account for scr refresh
                moneyImage_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyImage_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyImage_8.started')
                # update status
                moneyImage_8.status = STARTED
                moneyImage_8.setAutoDraw(True)
            
            # if moneyImage_8 is active this frame...
            if moneyImage_8.status == STARTED:
                # update params
                pass
            
            # *moneyText_8* updates
            
            # if moneyText_8 is starting this frame...
            if moneyText_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                moneyText_8.frameNStart = frameN  # exact frame index
                moneyText_8.tStart = t  # local t and not account for scr refresh
                moneyText_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moneyText_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moneyText_8.started')
                # update status
                moneyText_8.status = STARTED
                moneyText_8.setAutoDraw(True)
            
            # if moneyText_8 is active this frame...
            if moneyText_8.status == STARTED:
                # update params
                pass
            
            # *breakText* updates
            
            # if breakText is starting this frame...
            if breakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breakText.frameNStart = frameN  # exact frame index
                breakText.tStart = t  # local t and not account for scr refresh
                breakText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakText.started')
                # update status
                breakText.status = STARTED
                breakText.setAutoDraw(True)
            
            # if breakText is active this frame...
            if breakText.status == STARTED:
                # update params
                pass
            
            # *breakKey* updates
            waitOnFlip = False
            
            # if breakKey is starting this frame...
            if breakKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breakKey.frameNStart = frameN  # exact frame index
                breakKey.tStart = t  # local t and not account for scr refresh
                breakKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakKey, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakKey.started')
                # update status
                breakKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(breakKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(breakKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if breakKey.status == STARTED and not waitOnFlip:
                theseKeys = breakKey.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _breakKey_allKeys.extend(theseKeys)
                if len(_breakKey_allKeys):
                    breakKey.keys = _breakKey_allKeys[-1].name  # just the last key pressed
                    breakKey.rt = _breakKey_allKeys[-1].rt
                    breakKey.duration = _breakKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *curvedLine_10* updates
            
            # if curvedLine_10 is starting this frame...
            if curvedLine_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                curvedLine_10.frameNStart = frameN  # exact frame index
                curvedLine_10.tStart = t  # local t and not account for scr refresh
                curvedLine_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(curvedLine_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'curvedLine_10.started')
                # update status
                curvedLine_10.status = STARTED
                curvedLine_10.setAutoDraw(True)
            
            # if curvedLine_10 is active this frame...
            if curvedLine_10.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=break_screen,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_screen" ---
        for thisComponent in break_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for break_screen
        break_screen.tStop = globalClock.getTime(format='float')
        break_screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('break_screen.stopped', break_screen.tStop)
        # check responses
        if breakKey.keys in ['', [], None]:  # No response was made
            breakKey.keys = None
        trialLoop.addData('breakKey.keys',breakKey.keys)
        if breakKey.keys != None:  # we had a response
            trialLoop.addData('breakKey.rt', breakKey.rt)
            trialLoop.addData('breakKey.duration', breakKey.duration)
        # the Routine "break_screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrialLoop as finished
        if hasattr(thisTrialLoop, 'status'):
            thisTrialLoop.status = FINISHED
        # if awaiting a pause, pause now
        if trialLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trialLoop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trialLoop'
    trialLoop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "final_screen" ---
    # create an object to store info about Routine final_screen
    final_screen = data.Routine(
        name='final_screen',
        components=[bgImage_10, endText, endKey],
    )
    final_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for endKey
    endKey.keys = []
    endKey.rt = []
    _endKey_allKeys = []
    # store start times for final_screen
    final_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    final_screen.tStart = globalClock.getTime(format='float')
    final_screen.status = STARTED
    thisExp.addData('final_screen.started', final_screen.tStart)
    final_screen.maxDuration = None
    # keep track of which components have finished
    final_screenComponents = final_screen.components
    for thisComponent in final_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "final_screen" ---
    final_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *bgImage_10* updates
        
        # if bgImage_10 is starting this frame...
        if bgImage_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            bgImage_10.frameNStart = frameN  # exact frame index
            bgImage_10.tStart = t  # local t and not account for scr refresh
            bgImage_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bgImage_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bgImage_10.started')
            # update status
            bgImage_10.status = STARTED
            bgImage_10.setAutoDraw(True)
        
        # if bgImage_10 is active this frame...
        if bgImage_10.status == STARTED:
            # update params
            pass
        
        # *endText* updates
        
        # if endText is starting this frame...
        if endText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endText.frameNStart = frameN  # exact frame index
            endText.tStart = t  # local t and not account for scr refresh
            endText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endText.started')
            # update status
            endText.status = STARTED
            endText.setAutoDraw(True)
        
        # if endText is active this frame...
        if endText.status == STARTED:
            # update params
            pass
        
        # *endKey* updates
        waitOnFlip = False
        
        # if endKey is starting this frame...
        if endKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endKey.frameNStart = frameN  # exact frame index
            endKey.tStart = t  # local t and not account for scr refresh
            endKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endKey.started')
            # update status
            endKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endKey.status == STARTED and not waitOnFlip:
            theseKeys = endKey.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _endKey_allKeys.extend(theseKeys)
            if len(_endKey_allKeys):
                endKey.keys = _endKey_allKeys[0].name  # just the first key pressed
                endKey.rt = _endKey_allKeys[0].rt
                endKey.duration = _endKey_allKeys[0].duration
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=final_screen,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            final_screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in final_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "final_screen" ---
    for thisComponent in final_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for final_screen
    final_screen.tStop = globalClock.getTime(format='float')
    final_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('final_screen.stopped', final_screen.tStop)
    # check responses
    if endKey.keys in ['', [], None]:  # No response was made
        endKey.keys = None
    thisExp.addData('endKey.keys',endKey.keys)
    if endKey.keys != None:  # we had a response
        thisExp.addData('endKey.rt', endKey.rt)
        thisExp.addData('endKey.duration', endKey.duration)
    thisExp.nextEntry()
    # the Routine "final_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
