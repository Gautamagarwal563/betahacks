import React from 'react';
import {Composition} from 'remotion';
import {ConduitCinema, ConduitCinemaProps} from './ConduitCinema';

const INTRO_SEC = 1.8;
const OUTRO_SEC = 2.0;
const FPS = 30;

export const RemotionRoot: React.FC = () => (
  <Composition
    id="ConduitCinema"
    component={ConduitCinema}
    durationInFrames={FPS * 10}
    fps={FPS}
    width={1920}
    height={1080}
    defaultProps={{
      title: 'Conduit',
      subtitle: 'Directed by Conduit',
      phone: '+1 (443) 464-8118',
      shots: [],
    } satisfies ConduitCinemaProps}
    calculateMetadata={({props}) => {
      const shotsSec = (props.shots ?? []).reduce((s, x) => s + (x.duration ?? 5), 0);
      const total = INTRO_SEC + shotsSec + OUTRO_SEC;
      return {durationInFrames: Math.max(1, Math.round(total * FPS))};
    }}
  />
);
