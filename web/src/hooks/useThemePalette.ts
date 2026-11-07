import { useEffect, useState } from 'react';

export type ThemeMode = 'warm' | 'light' | 'dark';

export interface PlotThemePalette {
  mode: ThemeMode;
  plotBg: string;
  plotFg: string;
  labelFg: string;
  frameStroke: string;
  dotHalo: string;
  tipBg: string;
  tipStroke: string;
  cursorFill: string;
  cursorStroke: string;
}

const PALETTES: Record<ThemeMode, PlotThemePalette> = {
  warm: {
    mode: 'warm',
    plotBg: '#fbf8f1',
    plotFg: '#44403c',
    labelFg: '#57534e',
    frameStroke: '#ddd4c3',
    dotHalo: '#fbf8f1',
    tipBg: '#fffdf8',
    tipStroke: '#c7bda8',
    cursorFill: '#292524',
    cursorStroke: '#44403c',
  },
  light: {
    mode: 'light',
    plotBg: '#ffffff',
    plotFg: '#334155',
    labelFg: '#475569',
    frameStroke: '#e2e8f0',
    dotHalo: '#ffffff',
    tipBg: '#ffffff',
    tipStroke: '#cbd5e1',
    cursorFill: '#0f172a',
    cursorStroke: '#334155',
  },
  dark: {
    mode: 'dark',
    plotBg: '#111827',
    plotFg: '#cbd5e1',
    labelFg: '#94a3b8',
    frameStroke: '#243247',
    dotHalo: '#0b1120',
    tipBg: '#1e293b',
    tipStroke: '#475569',
    cursorFill: '#f8fafc',
    cursorStroke: '#cbd5e1',
  },
};

function readDocumentTheme(): ThemeMode {
  if (typeof document === 'undefined') return 'warm';
  const raw = document.documentElement.getAttribute('data-theme');
  if (raw === 'light' || raw === 'dark' || raw === 'warm') {
    return raw;
  }
  return 'warm';
}

export function useThemePalette(): PlotThemePalette {
  const [mode, setMode] = useState<ThemeMode>('warm');

  useEffect(() => {
    const sync = () => setMode(readDocumentTheme());
    sync();

    const observer = new MutationObserver(sync);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    window.addEventListener('gitltyn-theme-change', sync);

    return () => {
      observer.disconnect();
      window.removeEventListener('gitltyn-theme-change', sync);
    };
  }, []);

  return PALETTES[mode];
}
