import React from 'react';
import Header from './components/Header';
import Scene3D from './components/Scene3D';
import TextInput from './components/TextInput';
import Controls from './components/Controls';
import SequenceTimeline from './components/SequenceTimeline';
import InfoPanel from './components/InfoPanel';
import MocapStudio from './components/MocapStudio';
import { useAnimationController } from './engine/useAnimationController';

export default function App() {
  const controller = useAnimationController();
  const [mocapPose, setMocapPose] = React.useState(null);

  // Use mocap pose if it exists, otherwise use the sequence controller's pose
  const activePose = mocapPose || controller.currentPose;

  return (
    <div className="min-h-screen bg-surface relative overflow-hidden">
      {/* Animated background decorations */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] 
                      rounded-full bg-primary/5 blur-[120px] animate-pulse-slow" />
        <div className="absolute bottom-[-20%] left-[-10%] w-[500px] h-[500px] 
                      rounded-full bg-accent/5 blur-[100px] animate-pulse-slow"
             style={{ animationDelay: '2s' }} />
        <div className="absolute top-[30%] left-[20%] w-[300px] h-[300px] 
                      rounded-full bg-primary-dark/3 blur-[80px] animate-float" />
      </div>

      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="relative z-10 max-w-[1600px] w-full mx-auto px-4 sm:px-6 pb-2 h-auto lg:h-[calc(100vh-90px)] flex flex-col">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:gap-6 flex-1 h-full min-h-0">
          
          {/* Left Panel - Input, Controls & Timeline */}
          <div className="lg:col-span-4 xl:col-span-4 flex flex-col gap-4 order-2 lg:order-1 lg:overflow-y-auto hide-scrollbar pb-4 h-full">
            <TextInput
              onTranslate={controller.translateText}
              onReset={controller.reset}
              isPlaying={controller.isPlaying}
              inputText={controller.inputText}
            />

            <Controls
              isPlaying={controller.isPlaying}
              speed={controller.speed}
              progress={controller.progress}
              currentLabel={controller.currentLabel}
              currentIndex={controller.currentIndex}
              totalSteps={controller.sequence.length}
              onPlay={controller.play}
              onPause={controller.pause}
              onReset={controller.reset}
              onSpeedChange={controller.adjustSpeed}
            />

            <SequenceTimeline
              sequence={controller.sequence}
              currentIndex={controller.currentIndex}
            />

            <div className="hidden lg:flex flex-col gap-4 pb-6">
              <MocapStudio onPoseDetected={setMocapPose} />
              <InfoPanel />
            </div>
          </div>

          {/* Right - 3D Viewport */}
          <div className="lg:col-span-8 xl:col-span-8 flex flex-col order-1 lg:order-2 h-[60vh] lg:h-full min-h-0">
            <div className="glass rounded-2xl overflow-hidden glow-border relative flex-1 flex flex-col shadow-2xl" id="viewport-container">
              {/* Viewport Header */}
              <div className="absolute top-0 left-0 right-0 z-10 px-4 py-3 
                            bg-gradient-to-b from-surface/80 to-transparent">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${controller.isPlaying ? 'bg-success animate-pulse' : 'bg-text-muted'}`} />
                    <span className="text-xs text-text-secondary font-medium">
                      {controller.isPlaying ? 'Animating...' : 'Ready'}
                    </span>
                  </div>
                  <span className="text-[10px] text-text-muted font-mono">
                    3D Viewport • Drag to rotate
                  </span>
                </div>
              </div>

              {/* 3D Scene */}
              <div className="flex-1 w-full bg-gradient-to-b from-surface-light/30 to-surface/60 relative">
                <div className="absolute inset-0">
                  <Scene3D pose={activePose} />
                </div>
              </div>

              {/* Current Sign Label Overlay */}
              {controller.currentLabel && (
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
                  <div className="glass-strong px-6 py-3 rounded-2xl shadow-2xl">
                    <p className="text-xl font-bold text-text-primary text-center">
                      {controller.currentLabel}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 text-center py-2 px-4 h-[40px] flex items-center justify-center">
        <p className="text-xs text-text-muted">
          SignFlow — Real-time ASL Sign Language Translator • Built with React, Three.js & ❤️
        </p>
      </footer>
    </div>
  );
}
