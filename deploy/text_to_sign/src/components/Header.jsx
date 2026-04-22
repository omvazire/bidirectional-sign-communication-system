/**
 * App Header with branding and info
 */
import React from 'react';
import { Hand, Languages, Sparkles } from 'lucide-react';

export default function Header() {
  return (
    <header className="relative z-10" id="app-header">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        {/* Logo & Brand */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary via-primary-dark to-accent 
                          flex items-center justify-center shadow-lg shadow-primary/30 
                          animate-pulse-slow">
              <Hand className="w-6 h-6 text-white" />
            </div>
            <div className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full 
                          bg-success border-2 border-surface animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-display font-bold text-gradient tracking-tight">
              SignFlow
            </h1>
            <p className="text-[11px] text-text-muted tracking-wide">
              ASL Sign Language Translator
            </p>
          </div>
        </div>

        {/* Right badges */}
        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg 
                        bg-surface-light/50 border border-surface-lighter/30">
            <Languages className="w-3.5 h-3.5 text-accent" />
            <span className="text-xs text-text-secondary">ASL</span>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg 
                        bg-surface-light/50 border border-surface-lighter/30">
            <Sparkles className="w-3.5 h-3.5 text-warning" />
            <span className="text-xs text-text-secondary">Real-time</span>
          </div>
        </div>
      </div>
    </header>
  );
}
