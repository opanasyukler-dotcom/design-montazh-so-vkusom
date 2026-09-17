// VALERIA_REELS_STYLE.js
// Master motion preset for Reels

export const REELS_STYLE = {

  typography: {
    mainFont: "Manrope",
    subtitleWeight: 500,
    accentWeight: 800,

    subtitleCase: "normal",
    accentCase: "uppercase",

    handwrittenFont: "handwritten",

    subtitleMaxLines: 2,

    textPosition: {
      horizontal: "center",
      vertical: "lower-third"
    }
  },


  // -------------------------
  // GENERAL MOTION LANGUAGE
  // -------------------------

  motion: {
    style: "minimal-cinematic",

    defaultDuration: 0.35,

    easing: "cubic-bezier(0.22, 1, 0.36, 1)",

    avoid: [
      "bounce",
      "elastic",
      "spin",
      "random zoom",
      "aggressive transitions",
      "TikTok-template style",
      "excessive motion",
      "neon effects",
      "glitch"
    ]
  },


  // -------------------------
  // SUBTITLES
  // -------------------------

  subtitles: {

    animation: {
      type: "fade-up",

      from: {
        opacity: 0,
        translateY: 10,
        scale: 0.98
      },

      to: {
        opacity: 1,
        translateY: 0,
        scale: 1
      },

      duration: 0.28
    },

    // Highlight only meaningful words
    highlightRules: {
      maxWordsPerPhrase: 2,

      useFor: [
        "emotion",
        "important fact",
        "problem",
        "result",
        "key phrase"
      ]
    }
  },


  // -------------------------
  // ACCENT WORDS
  // -------------------------

  accentText: {

    style: {
      font: "Manrope",
      weight: 800,
      uppercase: true
    },

    animation: {
      scaleFrom: 0.92,
      opacityFrom: 0,
      duration: 0.25
    }
  },


  // -------------------------
  // HANDWRITTEN ELEMENTS
  // -------------------------

  handwriting: {

    useFor: [
      "short comments",
      "emotional remarks",
      "arrows",
      "underlines",
      "circles"
    ],

    maxCharacters: 20,

    animation: {
      type: "draw",
      duration: 0.5
    }
  },


  // -------------------------
  // ARROWS / MARKERS
  // -------------------------

  arrows: {

    style: "hand-drawn",

    lineWidth: 2,

    animation: {
      type: "stroke-draw",
      duration: 0.45
    },

    behavior: {
      pointToSubject: true,
      avoidFace: true,
      avoidImportantObjects: true
    }
  },


  // -------------------------
  // PHOTO / VIDEO INSERTS
  // -------------------------

  inserts: {

    fullscreen: false,

    maxScreenCoverage: 0.55,

    borderRadius: 18,

    animationIn: {
      opacity: [0, 1],
      scale: [0.96, 1],
      translateY: [12, 0],
      duration: 0.35
    },

    animationOut: {
      opacity: [1, 0],
      scale: [1, 0.98],
      duration: 0.25
    }
  },


  // -------------------------
  // CAMERA MOVEMENT
  // -------------------------

  camera: {

    allowDigitalZoom: true,

    zoomRange: [1.0, 1.06],

    useZoomFor: [
      "important sentence",
      "emotional moment",
      "transition to new thought"
    ],

    avoid: [
      "constant zooming",
      "fast punch zoom",
      "zoom on every sentence"
    ]
  },


  // -------------------------
  // TRANSITIONS
  // -------------------------

  transitions: {

    preferred: [
      "hard-cut",
      "match-cut",
      "soft-fade",
      "movement-cut"
    ],

    defaultDuration: 0.2,

    avoid: [
      "swipes",
      "3D transitions",
      "spins",
      "flash transitions",
      "template transitions"
    ]
  },


  // -------------------------
  // EDITING RHYTHM
  // -------------------------

  rhythm: {

    remove: [
      "long pauses",
      "unnecessary breaths",
      "um",
      "uh",
      "failed takes"
    ],

    keep: [
      "natural micro-pauses",
      "emotional reactions",
      "meaningful silence"
    ],

    principle:
      "Fast enough to retain attention but never hyperactive."
  },


  // -------------------------
  // MAIN RULE
  // -------------------------

  philosophy: `
  Animation must support the story rather than demonstrate editing.

  The final result should feel:
  cinematic,
  editorial,
  clean,
  contemporary,
  feminine,
  understated,
  natural.

  The viewer should notice the message first and the editing second.
  `
};
