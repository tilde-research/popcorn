'use client';

import {
  DEFAULT_LIVE_PORT,
  LIVE_PROTOCOL,
  appendLiveRow,
  liveEventsUrl,
  liveRecord,
  liveStatus,
  loadLivePort,
  probeLive,
  saveLivePort,
  validLivePort,
  type LiveProbe,
  type LiveRow,
  type LiveStatus,
} from '@/lib/live-bench';
import { useCallback, useEffect, useRef, useState } from 'react';

export type StreamState = 'idle' | 'checking' | 'opening' | 'open' | 'reconnecting' | 'complete';

export interface LiveBenchController {
  portText: string;
  probe: LiveProbe | null;
  streamState: StreamState;
  status: LiveStatus | null;
  rowsByOp: Record<string, LiveRow[]>;
  portError: string;
  changePort: (value: string) => void;
  check: () => Promise<void>;
}

function eventJson(event: Event): unknown {
  try {
    return JSON.parse((event as MessageEvent<string>).data) as unknown;
  } catch {
    return null;
  }
}

export function useLiveBench(): LiveBenchController {
  const [portText, setPortText] = useState(String(DEFAULT_LIVE_PORT));
  const [probe, setProbe] = useState<LiveProbe | null>(null);
  const [streamState, setStreamState] = useState<StreamState>('idle');
  const [connectedPort, setConnectedPort] = useState<number | null>(null);
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [rowsByOp, setRowsByOp] = useState<Record<string, LiveRow[]>>({});
  const [portError, setPortError] = useState('');
  const request = useRef(0);
  const sourceRef = useRef<EventSource | null>(null);
  const sessionRef = useRef('');

  const disconnect = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setConnectedPort(null);
  }, []);

  const changePort = useCallback(
    (value: string) => {
      request.current += 1;
      disconnect();
      sessionRef.current = '';
      setPortText(value);
      setProbe(null);
      setStatus(null);
      setRowsByOp({});
      setPortError('');
      setStreamState('idle');
    },
    [disconnect],
  );

  const scan = useCallback(async (port: number) => {
    const attempt = ++request.current;
    disconnect();
    saveLivePort(port);
    sessionRef.current = '';
    setPortError('');
    setProbe(null);
    setStatus(null);
    setRowsByOp({});
    setStreamState('checking');
    const result = await probeLive(port);
    if (request.current !== attempt) return;
    setProbe(result);
    if (result.kind === 'popcorn') {
      sessionRef.current = result.status.session;
      setStatus(result.status);
      if (result.status.state === 'running') {
        setConnectedPort(port);
        setStreamState('opening');
      } else {
        setStreamState('complete');
      }
    } else {
      setStatus(result.kind === 'incompatible' ? result.status : null);
      setStreamState('idle');
    }
  }, [disconnect]);

  const check = useCallback(async () => {
    const port = validLivePort(portText);
    if (port === null) {
      setPortError('Enter a port from 1 to 65535.');
      return;
    }
    await scan(port);
  }, [portText, scan]);

  useEffect(() => {
    const port = loadLivePort();
    setPortText(String(port));
    void scan(port);
  }, [scan]);

  useEffect(() => {
    if (connectedPort === null) return;
    let source: EventSource;
    try {
      source = new EventSource(liveEventsUrl(connectedPort));
    } catch {
      setStreamState('reconnecting');
      return;
    }
    sourceRef.current = source;

    const updateStatus = (event: Event) => {
      const next = liveStatus(eventJson(event));
      if (next === null) return;
      if (sessionRef.current && sessionRef.current !== next.session) setRowsByOp({});
      sessionRef.current = next.session;
      setStatus(next);
      setProbe({ kind: next.protocol === LIVE_PROTOCOL ? 'popcorn' : 'incompatible', status: next });
    };
    const addRecord = (event: Event) => {
      const message = event as MessageEvent<string>;
      const record = liveRecord(eventJson(event));
      const sequence = Number(message.lastEventId);
      if (record === null || !Number.isInteger(sequence)) return;
      setRowsByOp((current) => ({
        ...current,
        [record.op]: appendLiveRow(current[record.op] ?? [], { sequence, record }, 512),
      }));
    };
    const complete = (event: Event) => {
      updateStatus(event);
      source.close();
      sourceRef.current = null;
      setConnectedPort(null);
      setStreamState('complete');
    };

    source.addEventListener('session', updateStatus);
    source.addEventListener('progress', updateStatus);
    source.addEventListener('record', addRecord);
    source.addEventListener('complete', complete);
    source.onopen = () => setStreamState('open');
    source.onerror = () => setStreamState('reconnecting');

    return () => {
      source.close();
      if (sourceRef.current === source) sourceRef.current = null;
    };
  }, [connectedPort]);

  useEffect(
    () => () => {
      request.current += 1;
      sourceRef.current?.close();
    },
    [],
  );

  return { portText, probe, streamState, status, rowsByOp, portError, changePort, check };
}
