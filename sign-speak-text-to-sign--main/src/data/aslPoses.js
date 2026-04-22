// ASL Hand Pose Data - Accurate finger joint angles for ASL signs
// Each finger: { base, mid, tip, spread } (radians)
import { ACTUAL_POSES } from './actualPoses';
const f = (base=0,mid=0,tip=0,spread=0) => ({base,mid,tip,spread});
const STRAIGHT = f(0,0,0,0);
const CURLED = f(1.5,1.5,1.2,0);
const HALF = f(0.8,0.9,0.7,0);
const HOOK = f(0,1.4,1.4,0);
const SLIGHT = f(0.3,0.2,0.1,0);

// === REUSABLE HAND SHAPES ===
const FIST={thumb:f(0.5,0.3,0,-0.1),index:CURLED,middle:CURLED,ring:CURLED,pinky:CURLED};
const FLAT={thumb:f(0.8,0.4,0,-0.3),index:STRAIGHT,middle:STRAIGHT,ring:STRAIGHT,pinky:STRAIGHT};
const OPEN={thumb:f(0,0,0,0.5),index:f(0,0,0,0.15),middle:f(0,0,0,0.05),ring:f(0,0,0,-0.05),pinky:f(0,0,0,-0.2)};
const C_SHAPE={thumb:f(0.3,0.2,0.1,0.3),index:f(0.5,0.4,0.3,0.1),middle:f(0.5,0.4,0.3,0),ring:f(0.5,0.4,0.3,-0.05),pinky:f(0.5,0.4,0.3,-0.15)};
const O_SHAPE={thumb:f(0.5,0.4,0.3,0.1),index:f(0.9,0.8,0.5,0.05),middle:f(0.9,0.8,0.5,0),ring:f(0.9,0.8,0.5,-0.05),pinky:f(0.9,0.8,0.5,-0.1)};
const POINT={thumb:f(0.5,0.2,0,-0.3),index:STRAIGHT,middle:CURLED,ring:CURLED,pinky:CURLED};
const V_SHAPE={thumb:f(0.5,0.2,0,-0.3),index:f(0,0,0,0.2),middle:f(0,0,0,-0.2),ring:CURLED,pinky:CURLED};
const CLAW={thumb:f(0.2,0.2,0,0.4),index:f(0.4,0.7,0.6,0.15),middle:f(0.4,0.7,0.6,0.05),ring:f(0.4,0.7,0.6,-0.05),pinky:f(0.4,0.7,0.6,-0.2)};
const Y_SHAPE={thumb:f(0,0,0,0.7),index:CURLED,middle:CURLED,ring:CURLED,pinky:f(0,0,0,-0.3)};
const L_SHAPE={thumb:f(0,0,0,0.7),index:STRAIGHT,middle:CURLED,ring:CURLED,pinky:CURLED};
const ILY={thumb:f(0,0,0,0.7),index:f(0,0,0,0.2),middle:CURLED,ring:CURLED,pinky:f(0,0,0,-0.3)};
const K_SHAPE={thumb:f(0.3,0,0,0.3),index:f(0,0,0,0.15),middle:f(0.3,0,0,-0.15),ring:CURLED,pinky:CURLED};
const X_SHAPE={thumb:f(0.4,0.2,0,-0.2),index:HOOK,middle:CURLED,ring:CURLED,pinky:CURLED};
const D_SHAPE={thumb:f(0.6,0.3,0,-0.2),index:STRAIGHT,middle:CURLED,ring:CURLED,pinky:CURLED};
const W_SHAPE={thumb:f(0.5,0.2,0,-0.3),index:f(0,0,0,0.2),middle:STRAIGHT,ring:f(0,0,0,-0.2),pinky:CURLED};
const R_SHAPE={thumb:f(0.5,0.2,0,-0.2),index:f(0,0,0,0.05),middle:f(0,0,0,-0.05),ring:CURLED,pinky:CURLED};
const U_SHAPE={thumb:f(0.5,0.2,0,-0.3),index:f(0,0,0,0.03),middle:f(0,0,0,-0.03),ring:CURLED,pinky:CURLED};
const F_SHAPE={thumb:f(0.5,0.5,0.3,0),index:f(1.4,1.2,0.8,0),middle:STRAIGHT,ring:STRAIGHT,pinky:f(0,0,0,0.1)};

const w=(x=0,y=0,z=0)=>({x,y,z});

export const ASL_POSES = {
  A:{...FIST,thumb:f(0.2,0,0,0.6),wrist:w()},
  B:{...FLAT,wrist:w()},
  C:{...C_SHAPE,wrist:w()},
  D:{...D_SHAPE,wrist:w()},
  E:{thumb:f(0.6,0.3,0,-0.2),index:f(1.2,0.5,0,0),middle:f(1.2,0.5,0,0),ring:f(1.2,0.5,0,0),pinky:f(1.2,0.5,0,0),wrist:w()},
  F:{...F_SHAPE,wrist:w()},
  G:{...{thumb:f(0.1,0,0,0.4),index:STRAIGHT,middle:CURLED,ring:CURLED,pinky:CURLED},wrist:w(0,-1.2,0)},
  H:{thumb:f(0.4,0.2,0,-0.2),index:STRAIGHT,middle:STRAIGHT,ring:CURLED,pinky:CURLED,wrist:w(0,-1.2,0)},
  I:{thumb:f(0.4,0.2,0,-0.2),index:CURLED,middle:CURLED,ring:CURLED,pinky:STRAIGHT,wrist:w()},
  J:{thumb:f(0.4,0.2,0,-0.2),index:CURLED,middle:CURLED,ring:CURLED,pinky:STRAIGHT,wrist:w(0,0,0.3)},
  K:{...K_SHAPE,wrist:w()},
  L:{...L_SHAPE,wrist:w()},
  M:{thumb:f(0.6,0.2,0,-0.4),index:CURLED,middle:CURLED,ring:CURLED,pinky:CURLED,wrist:w()},
  N:{thumb:f(0.5,0.2,0,-0.3),index:CURLED,middle:CURLED,ring:CURLED,pinky:CURLED,wrist:w()},
  O:{...O_SHAPE,wrist:w()},
  P:{thumb:f(0.2,0,0,0.3),index:f(0.3,0,0,0.1),middle:f(0.8,0.5,0.3,-0.1),ring:CURLED,pinky:CURLED,wrist:w(0.8,-0.6,0)},
  Q:{thumb:f(0.3,0.2,0,0.2),index:f(0.5,0.2,0,0),middle:CURLED,ring:CURLED,pinky:CURLED,wrist:w(1,0,0)},
  R:{...R_SHAPE,wrist:w()},
  S:{...FIST,thumb:f(0.5,0.3,0,-0.1),wrist:w()},
  T:{thumb:f(0.3,0.2,0,0.1),index:CURLED,middle:CURLED,ring:CURLED,pinky:CURLED,wrist:w()},
  U:{...U_SHAPE,wrist:w()},
  V:{...V_SHAPE,wrist:w()},
  W:{...W_SHAPE,wrist:w()},
  X:{...X_SHAPE,wrist:w()},
  Y:{...Y_SHAPE,wrist:w()},
  Z:{...POINT,wrist:w()},
  '0':{...O_SHAPE,wrist:w()},
  '1':{...POINT,wrist:w()},
  '2':{...V_SHAPE,wrist:w()},
  '3':{thumb:f(0,0,0,0.5),index:f(0,0,0,0.15),middle:f(0,0,0,-0.15),ring:CURLED,pinky:CURLED,wrist:w()},
  '4':{thumb:f(0.8,0.4,0,-0.3),index:f(0,0,0,0.15),middle:f(0,0,0,0.05),ring:f(0,0,0,-0.05),pinky:f(0,0,0,-0.15),wrist:w()},
  '5':{...OPEN,wrist:w()},
  '6':{thumb:f(0.5,0.4,0.3,0),index:STRAIGHT,middle:STRAIGHT,ring:STRAIGHT,pinky:CURLED,wrist:w()},
  '7':{thumb:f(0.5,0.4,0.3,0),index:STRAIGHT,middle:STRAIGHT,ring:CURLED,pinky:STRAIGHT,wrist:w()},
  '8':{thumb:f(0.5,0.4,0.3,0),index:STRAIGHT,middle:CURLED,ring:STRAIGHT,pinky:STRAIGHT,wrist:w()},
  '9':{...F_SHAPE,wrist:w()},
  ' ':{thumb:f(0.1,0,0,0.3),index:f(0.15,0.08,0.03,0.08),middle:f(0.15,0.08,0.03,0),ring:f(0.15,0.08,0.03,-0.04),pinky:f(0.15,0.08,0.03,-0.12),wrist:w()},
  ...ACTUAL_POSES,
};

// Helper to make a pose from a shape + wrist
const pose=(shape,wrist=w(),dur=600)=>({...shape,wrist,duration:dur});

export const WORD_SIGNS = {
  // === GREETINGS ===
  hello:{description:'Open hand wave',poses:[pose(OPEN,w(0,0,0.3),350),pose(OPEN,w(0,0,-0.3),350),pose(OPEN,w(0,0,0.2),300)]},
  hi:{description:'Open hand wave',poses:[pose(OPEN,w(0,0,0.3),300),pose(OPEN,w(0,0,-0.3),300)]},
  bye:{description:'Open hand wave goodbye',poses:[pose(OPEN,w(0,0,0.2),300),pose(FLAT,w(0,0,-0.1),300),pose(OPEN,w(0,0,0.2),300),pose(FLAT,w(0,0,-0.1),300)]},
  goodbye:{description:'Wave goodbye',poses:[pose(OPEN,w(0,0,0.2),300),pose(FLAT,w(0,0,-0.1),300),pose(OPEN,w(0,0,0.2),300)]},
  welcome:{description:'Open hand gesture inward',poses:[pose(FLAT,w(0,0.3,0),400),pose(FLAT,w(-0.2,-0.2,0),400)]},
  // === COMMON SOCIAL ===
  please:{description:'Flat hand circles on chest',poses:[pose(FLAT,w(0,0.2,0.2),450),pose(FLAT,w(0,-0.2,-0.2),450),pose(FLAT,w(0,0.2,0.2),400)]},
  thanks:{description:'Flat hand from chin forward',poses:[pose(FLAT,w(-0.3,0,0),500),pose(FLAT,w(0.3,0,0),500)]},
  thank:{description:'Flat hand from chin forward',poses:[pose(FLAT,w(-0.3,0,0),500),pose(FLAT,w(0.3,0,0),500)]},
  sorry:{description:'Fist circles on chest',poses:[pose(FIST,w(0,0.2,0.2),400),pose(FIST,w(0,-0.2,-0.2),400),pose(FIST,w(0,0.2,0.2),400)]},
  excuse:{description:'Fingertips brush across palm',poses:[pose(FLAT,w(0,0.1,0),400),pose(FLAT,w(0.2,-0.1,0),400)]},
  congratulations:{description:'Clapping motion',poses:[pose(CLAW,w(0,0.2,0),400),pose(CLAW,w(0,-0.2,0),400),pose(CLAW,w(0,0.2,0),400)]},
  // === RESPONSES ===
  yes:{description:'Fist nods like head nodding',poses:[pose(FIST,w(-0.4,0,0),350),pose(FIST,w(0.2,0,0),350),pose(FIST,w(-0.4,0,0),350)]},
  no:{description:'Index+middle snap to thumb',poses:[pose({thumb:f(0.2,0,0,0.3),index:f(0,0,0,0.1),middle:f(0,0,0,-0.1),ring:CURLED,pinky:CURLED},w(),300),pose({thumb:f(0.5,0.3,0.2,0),index:f(0.5,0.3,0.2,0),middle:f(0.5,0.3,0.2,0),ring:CURLED,pinky:CURLED},w(),300)]},
  maybe:{description:'Flat hands tilt side to side',poses:[pose(FLAT,w(0,0,0.3),350),pose(FLAT,w(0,0,-0.3),350),pose(FLAT,w(0,0,0.2),300)]},
  ok:{description:'OK sign - thumb index circle',poses:[pose(F_SHAPE,w(),900)]},
  right:{description:'Index on index - correct',poses:[pose(POINT,w(0,-0.3,0),500),pose(POINT,w(0,-0.1,0),400)]},
  wrong:{description:'Y hand on chin',poses:[pose(Y_SHAPE,w(-0.2,0,0),700)]},
  // === PRONOUNS ===
  me:{description:'Point to self',poses:[pose(POINT,w(-0.3,0,0),700)]},
  i:{description:'Point to self',poses:[pose(POINT,w(-0.3,0,0),700)]},
  you:{description:'Point forward',poses:[pose(POINT,w(0.3,-0.4,0),700)]},
  he:{description:'Point to side',poses:[pose(POINT,w(0,0.4,0),600)]},
  she:{description:'Point to side',poses:[pose(POINT,w(0,0.4,0),600)]},
  they:{description:'Point and sweep',poses:[pose(POINT,w(0,0.3,0),400),pose(POINT,w(0,-0.3,0),400)]},
  we:{description:'Point self then sweep',poses:[pose(POINT,w(-0.2,0.2,0),400),pose(POINT,w(-0.2,-0.3,0),400)]},
  my:{description:'Flat hand on chest',poses:[pose(FLAT,w(-0.2,0,0),700)]},
  your:{description:'Flat hand toward person',poses:[pose(FLAT,w(0.3,0,0),700)]},
  // === QUESTION WORDS ===
  what:{description:'Palms up, shake',poses:[pose(OPEN,w(0.3,0,0),350),pose(OPEN,w(0.3,0,0.2),300),pose(OPEN,w(0.3,0,-0.2),300)]},
  where:{description:'Index wags side to side',poses:[pose(POINT,w(0,0,0.3),300),pose(POINT,w(0,0,-0.3),300),pose(POINT,w(0,0,0.2),300)]},
  when:{description:'Index circles then touches',poses:[pose(POINT,w(0,0.2,0),400),pose(POINT,w(0,-0.2,0),400)]},
  who:{description:'Thumb on chin, index curls',poses:[pose(L_SHAPE,w(-0.2,0,0),400),pose(X_SHAPE,w(-0.2,0,0),400)]},
  why:{description:'Touch forehead, Y hand',poses:[pose(OPEN,w(-0.2,0,0),400),pose(Y_SHAPE,w(0,0,0),500)]},
  how:{description:'Knuckles together, roll out',poses:[pose(FIST,w(0,0,-0.2),400),pose(OPEN,w(0.2,0,0),400)]},
  // === COMMON VERBS ===
  want:{description:'Claw hands pull toward self',poses:[pose(CLAW,w(0.2,0,0),400),pose(CLAW,w(-0.2,0,0),400)]},
  need:{description:'X hand bends down',poses:[pose(X_SHAPE,w(0,0,0),400),pose(X_SHAPE,w(0.4,0,0),400)]},
  like:{description:'Thumb+middle from chest pull away',poses:[pose({thumb:f(0.2,0,0,0.3),index:CURLED,middle:f(0.3,0,0,0),ring:CURLED,pinky:CURLED},w(-0.2,0,0),400),pose({thumb:f(0.2,0,0,0.5),index:CURLED,middle:STRAIGHT,ring:CURLED,pinky:CURLED},w(0.2,0,0),400)]},
  love:{description:'ILY sign - thumb index pinky',poses:[pose(ILY,w(),1000)]},
  hate:{description:'Flick middle fingers away',poses:[pose({thumb:f(0.3,0,0,0.2),index:STRAIGHT,middle:f(0.3,0,0,0),ring:CURLED,pinky:CURLED},w(),400),pose(OPEN,w(0.2,0,0),400)]},
  know:{description:'Flat hand taps forehead',poses:[pose(FLAT,w(-0.3,0,0),400),pose(FLAT,w(-0.15,0,0),300),pose(FLAT,w(-0.3,0,0),300)]},
  think:{description:'Index points to forehead',poses:[pose(POINT,w(-0.3,0,0),700)]},
  feel:{description:'Middle finger brushes up chest',poses:[pose({...OPEN,middle:f(0.3,0,0,0)},w(-0.2,0,0),400),pose({...OPEN,middle:f(0.3,0,0,0)},w(-0.4,0,0),400)]},
  see:{description:'V from eyes forward',poses:[pose(V_SHAPE,w(-0.3,0,0),400),pose(V_SHAPE,w(0.1,0,0),400)]},
  look:{description:'V hand points from eyes',poses:[pose(V_SHAPE,w(-0.2,0,0),400),pose(V_SHAPE,w(0.2,0,0),400)]},
  hear:{description:'Index points to ear',poses:[pose(POINT,w(-0.2,0.3,0),700)]},
  listen:{description:'Cupped hand to ear',poses:[pose(C_SHAPE,w(-0.2,0.3,0),700)]},
  say:{description:'Index circles from mouth',poses:[pose(POINT,w(-0.3,0,0),400),pose(POINT,w(-0.1,0,0.2),400)]},
  tell:{description:'Index from chin forward',poses:[pose(POINT,w(-0.3,0,0),400),pose(POINT,w(0.2,0,0),400)]},
  ask:{description:'Index pulls back in hook',poses:[pose(POINT,w(0.2,0,0),400),pose(X_SHAPE,w(-0.1,0,0),400)]},
  understand:{description:'Index flicks up at forehead',poses:[pose(FIST,w(-0.3,0,0),400),pose(POINT,w(-0.3,0,0),400)]},
  learn:{description:'Grab from palm to forehead',poses:[pose(OPEN,w(0.2,0,0),400),pose(O_SHAPE,w(-0.3,0,0),400)]},
  teach:{description:'O hands move forward from head',poses:[pose(O_SHAPE,w(-0.3,0,0),400),pose(O_SHAPE,w(0.2,0,0),400)]},
  help:{description:'Fist on flat palm lifts up',poses:[pose(FIST,w(0.2,0,0),400),pose(FIST,w(-0.3,0,0),400)]},
  try:{description:'T hands push forward',poses:[pose({...FIST,thumb:f(0.3,0.2,0,0.1)},w(0,0,0),400),pose({...FIST,thumb:f(0.3,0.2,0,0.1)},w(0.3,0,0),400)]},
  give:{description:'O hand moves forward',poses:[pose(O_SHAPE,w(-0.1,0,0),400),pose(OPEN,w(0.3,0,0),400)]},
  take:{description:'Grab from air, pull in',poses:[pose(OPEN,w(0.2,0,0),400),pose(FIST,w(-0.1,0,0),400)]},
  have:{description:'Bent hands touch chest',poses:[pose(FLAT,w(-0.2,0,0),600)]},
  make:{description:'Fists twist together',poses:[pose(FIST,w(0,0,0.2),400),pose(FIST,w(0,0,-0.2),400)]},
  do:{description:'C hands move side to side',poses:[pose(C_SHAPE,w(0,0,0.2),400),pose(C_SHAPE,w(0,0,-0.2),400)]},
  go:{description:'Index points forward, moves out',poses:[pose(POINT,w(0,0,0),400),pose(POINT,w(0.3,-0.3,0),400)]},
  come:{description:'Index beckons toward self',poses:[pose(POINT,w(0.2,0,0),400),pose(POINT,w(-0.2,0,0),400)]},
  walk:{description:'Flat hands alternate stepping',poses:[pose(FLAT,w(0,0,0.15),350),pose(FLAT,w(0.15,0,-0.15),350),pose(FLAT,w(0,0,0.15),350)]},
  run:{description:'L hands, index hooks while moving',poses:[pose(L_SHAPE,w(0,0,0),350),pose(X_SHAPE,w(0.2,0,0),350),pose(L_SHAPE,w(0.1,0,0),300)]},
  sit:{description:'Bent V sits on fingers',poses:[pose(V_SHAPE,w(0.3,0,0),400),pose({...V_SHAPE,index:f(0.5,0.4,0,0.2),middle:f(0.5,0.4,0,-0.2)},w(0.3,0,0),500)]},
  stand:{description:'V stands on flat palm',poses:[pose(V_SHAPE,w(0.5,0,0),700)]},
  wait:{description:'Open hands wiggle fingers',poses:[pose(OPEN,w(0.2,0,0),400),pose(CLAW,w(0.2,0,0),300),pose(OPEN,w(0.2,0,0),300),pose(CLAW,w(0.2,0,0),300)]},
  stop:{description:'Flat hand chops down on palm',poses:[pose(FLAT,w(-0.5,0,0),350),pose(FLAT,w(0.1,0,0),350)]},
  start:{description:'Index twists between V fingers',poses:[pose(POINT,w(0,0,0.2),400),pose(POINT,w(0,0,-0.2),400)]},
  finish:{description:'Open hands flip outward',poses:[pose(OPEN,w(0,0,0),400),pose(OPEN,w(0.3,0.3,0),400)]},
  open:{description:'Flat hands open like book',poses:[pose(FLAT,w(0,0,-0.2),400),pose(FLAT,w(0,0,0.3),400)]},
  close:{description:'Flat hands close together',poses:[pose(FLAT,w(0,0,0.2),400),pose(FLAT,w(0,0,-0.2),400)]},
  eat:{description:'Flat O taps mouth repeatedly',poses:[pose(O_SHAPE,w(-0.2,0,0),350),pose(O_SHAPE,w(-0.05,0,0),300),pose(O_SHAPE,w(-0.2,0,0),300)]},
  drink:{description:'C hand tips toward mouth',poses:[pose(C_SHAPE,w(0,0,0),400),pose(C_SHAPE,w(-0.5,0,0),400)]},
  sleep:{description:'Open hand draws down face',poses:[pose(OPEN,w(-0.2,0,0),400),pose(O_SHAPE,w(0,0,0),500)]},
  wake:{description:'Closed eyes open at eyes',poses:[pose(FIST,w(-0.3,0,0),400),pose(L_SHAPE,w(-0.3,0,0),400)]},
  work:{description:'Fist taps on fist',poses:[pose(FIST,w(0,0,0.2),350),pose(FIST,w(0.15,0,-0.2),350),pose(FIST,w(0,0,0.2),300)]},
  play:{description:'Y hands shake',poses:[pose(Y_SHAPE,w(0,0,0.2),350),pose(Y_SHAPE,w(0,0,-0.2),350),pose(Y_SHAPE,w(0,0,0.2),300)]},
  read:{description:'V scans across palm',poses:[pose(V_SHAPE,w(0.2,0.2,0),400),pose(V_SHAPE,w(0.2,-0.2,0),400)]},
  write:{description:'Pinch hand writes on palm',poses:[pose(O_SHAPE,w(0.2,0.2,0),350),pose(O_SHAPE,w(0.2,0,0),300),pose(O_SHAPE,w(0.2,-0.2,0),300)]},
  drive:{description:'Fists turn steering wheel',poses:[pose(FIST,w(0,0,0.3),400),pose(FIST,w(0,0,-0.3),400)]},
  buy:{description:'Hand lifts from palm forward',poses:[pose(FLAT,w(0.2,0,0),400),pose(FLAT,w(-0.1,0,0),400)]},
  pay:{description:'Index flicks off palm',poses:[pose(POINT,w(0.2,0,0),400),pose(POINT,w(0.4,0,0),400)]},
  call:{description:'Y hand at ear',poses:[pose(Y_SHAPE,w(-0.2,0,0),700)]},
  meet:{description:'Index fingers come together',poses:[pose(POINT,w(0,0,0.3),400),pose(POINT,w(0,0,0),400)]},
  practice:{description:'A hand rubs back and forth',poses:[pose(FIST,w(0,0,0.2),350),pose(FIST,w(0,0,-0.2),350),pose(FIST,w(0,0,0.2),300)]},
  remember:{description:'Thumb from forehead to fist',poses:[pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.3,0,0),400),pose(FIST,w(-0.1,0,0),400)]},
  forget:{description:'Hand wipes across forehead',poses:[pose(OPEN,w(-0.3,0.2,0),400),pose(FIST,w(-0.3,-0.2,0),400)]},
  // === ADJECTIVES ===
  good:{description:'Flat hand from chin forward',poses:[pose(FLAT,w(-0.2,0,0),400),pose(FLAT,w(0.3,0,0),400)]},
  bad:{description:'Flat hand from chin down',poses:[pose(FLAT,w(-0.2,0,0),400),pose(FLAT,w(0.5,0,0),400)]},
  big:{description:'Hands spread wide apart',poses:[pose(L_SHAPE,w(0,0,-0.2),400),pose(L_SHAPE,w(0,0,0.4),400)]},
  small:{description:'Flat hands close together',poses:[pose(FLAT,w(0,0,0.2),400),pose(FLAT,w(0,0,0),400)]},
  little:{description:'Flat hands show small space',poses:[pose(FLAT,w(0,0,0.1),400),pose(FLAT,w(0,0,-0.05),400)]},
  happy:{description:'Flat hand brushes up on chest',poses:[pose(FLAT,w(0,0,0),400),pose(FLAT,w(-0.3,0,0),400)]},
  sad:{description:'Open hands drop down face',poses:[pose(OPEN,w(-0.2,0,0),400),pose(OPEN,w(0.3,0,0),500)]},
  angry:{description:'Claw hand pulls from face',poses:[pose(CLAW,w(-0.3,0,0),400),pose(CLAW,w(0,0,0),400)]},
  tired:{description:'Bent hands drop on chest',poses:[pose(FLAT,w(-0.2,0,0),400),pose(HALF,w(0.1,0,0),500)]},
  sick:{description:'Middle finger taps forehead',poses:[pose({...OPEN,middle:f(0.3,0,0,0)},w(-0.3,0,0),500),pose({...OPEN,middle:f(0.3,0,0,0)},w(-0.15,0,0),400)]},
  hungry:{description:'C hand pulls down from throat',poses:[pose(C_SHAPE,w(-0.3,0,0),400),pose(C_SHAPE,w(0,0,0),400)]},
  thirsty:{description:'Index draws down throat',poses:[pose(POINT,w(-0.3,0,0),400),pose(POINT,w(-0.1,0,0),400)]},
  hot:{description:'Claw hand turns from mouth out',poses:[pose(CLAW,w(-0.2,0,0),400),pose(CLAW,w(0.2,0.2,0),400)]},
  cold:{description:'Fists shake like shivering',poses:[pose(FIST,w(0,0,0.15),300),pose(FIST,w(0,0,-0.15),300),pose(FIST,w(0,0,0.15),300)]},
  fast:{description:'Index pulls back quickly',poses:[pose(L_SHAPE,w(0,0,0),300),pose(X_SHAPE,w(0,0,0),250)]},
  slow:{description:'Hand draws slowly up arm',poses:[pose(OPEN,w(0.2,0.2,0),500),pose(OPEN,w(-0.1,-0.1,0),500)]},
  easy:{description:'Bent hand brushes up on fingers',poses:[pose(FLAT,w(0,0,0),400),pose(FLAT,w(-0.2,0,0),400)]},
  hard:{description:'Bent V strikes fist',poses:[pose(V_SHAPE,w(-0.2,0,0),400),pose(V_SHAPE,w(0.2,0,0),300)]},
  new:{description:'Curved hand scoops across palm',poses:[pose(FLAT,w(0,0.2,0),400),pose(FLAT,w(0,-0.2,0),400)]},
  old:{description:'C hand drops from chin',poses:[pose(C_SHAPE,w(-0.2,0,0),400),pose(FIST,w(0,0,0),400)]},
  beautiful:{description:'Open hand circles face',poses:[pose(OPEN,w(-0.2,0.2,0),400),pose(OPEN,w(-0.2,-0.2,0),400),pose(O_SHAPE,w(-0.1,0,0),400)]},
  clean:{description:'Flat hand wipes across palm',poses:[pose(FLAT,w(0,0.2,0),400),pose(FLAT,w(0,-0.2,0),400)]},
  dirty:{description:'Hand wiggles under chin',poses:[pose(OPEN,w(-0.2,0,0),350),pose(CLAW,w(-0.2,0,0),350)]},
  quiet:{description:'Index crosses lips, hands down',poses:[pose(POINT,w(-0.2,0,0),400),pose(FLAT,w(0.2,0,0),400)]},
  loud:{description:'Point to ear, shake fists',poses:[pose(POINT,w(-0.2,0.3,0),350),pose(FIST,w(0,0,0.2),300),pose(FIST,w(0,0,-0.2),300)]},
  strong:{description:'Flex muscle',poses:[pose(FIST,w(-0.3,0,0),400),pose(FIST,w(-0.5,0,0),400)]},
  important:{description:'F hands rise to center',poses:[pose(F_SHAPE,w(0.2,0,0),400),pose(F_SHAPE,w(-0.2,0,0),400)]},
  different:{description:'Index fingers cross and separate',poses:[pose(POINT,w(0,0,0),400),pose(POINT,w(0,0,0.4),400)]},
  same:{description:'Y hand moves together',poses:[pose(Y_SHAPE,w(0,0,0.2),400),pose(Y_SHAPE,w(0,0,-0.1),400)]},
  more:{description:'Flat O hands tap together',poses:[pose(O_SHAPE,w(0,0,0.15),350),pose(O_SHAPE,w(0,0,-0.1),300),pose(O_SHAPE,w(0,0,0.15),300)]},
  // === TIME ===
  now:{description:'Bent hands drop together',poses:[pose(FLAT,w(0,0,0),350),pose(FLAT,w(0.3,0,0),350)]},
  today:{description:'Now + day - hands drop',poses:[pose(FLAT,w(0,0,0),400),pose(FLAT,w(0.3,0,0),400)]},
  tomorrow:{description:'Thumb on cheek moves forward',poses:[pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...FIST,thumb:f(0,0,0,0.5)},w(0.2,0,0),400)]},
  yesterday:{description:'Thumb touches cheek then jaw',poses:[pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.3,0.2,0),400)]},
  morning:{description:'Flat hand rises like sun',poses:[pose(FLAT,w(0.3,0,0),400),pose(FLAT,w(-0.2,0,0),400)]},
  evening:{description:'Hand drops like setting sun',poses:[pose(FLAT,w(-0.2,0,0),400),pose(FLAT,w(0.3,0,0),400)]},
  night:{description:'Bent hand covers horizon',poses:[pose(FLAT,w(0,0,0),400),pose(FLAT,w(0.4,0,0),400)]},
  week:{description:'Index slides across palm',poses:[pose(POINT,w(0,0.2,0),400),pose(POINT,w(0,-0.2,0),400)]},
  month:{description:'Index slides down finger',poses:[pose(POINT,w(-0.2,0,0),400),pose(POINT,w(0.2,0,0),400)]},
  year:{description:'Fists circle each other',poses:[pose(FIST,w(0,0,0.2),400),pose(FIST,w(0,0,-0.2),400),pose(FIST,w(0,0,0.2),300)]},
  always:{description:'Index circles forward',poses:[pose(POINT,w(0,0.2,0),350),pose(POINT,w(0.2,0,0),350),pose(POINT,w(0,-0.2,0),350)]},
  never:{description:'Flat hand zigzags down',poses:[pose(FLAT,w(-0.2,0,0.2),300),pose(FLAT,w(0,0,-0.2),300),pose(FLAT,w(0.2,0,0.2),300)]},
  sometimes:{description:'Index touches palm periodically',poses:[pose(POINT,w(0.2,0,0),350),pose(POINT,w(0,0,0),300),pose(POINT,w(0.2,0,0),300)]},
  soon:{description:'F hand near mouth twists',poses:[pose(F_SHAPE,w(-0.2,0,0),400),pose(F_SHAPE,w(-0.2,0,0.3),400)]},
  later:{description:'L hand thumb turns forward',poses:[pose(L_SHAPE,w(0,0,0),400),pose(L_SHAPE,w(0.3,0,0),400)]},
  again:{description:'Bent hand flips into palm',poses:[pose(FLAT,w(0.3,0,0),400),pose(FLAT,w(0,0,0),400)]},
  // === PEOPLE ===
  family:{description:'F hands circle outward',poses:[
    { right: pose(F_SHAPE,w(0,0,0.2),400), left: pose(F_SHAPE,w(0,0,-0.2),400) },
    { right: pose(F_SHAPE,w(0,0,-0.2),400), left: pose(F_SHAPE,w(0,0,0.2),400) },
    { right: pose(F_SHAPE,w(0,0,0.2),300), left: pose(F_SHAPE,w(0,0,-0.2),300) }
  ]},
  mother:{description:'Open hand thumb taps chin',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.1,0,0),300),pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.2,0,0),300)]},
  mom:{description:'Open hand thumb taps chin',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.1,0,0),300)]},
  father:{description:'Open hand thumb taps forehead',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.35,0,0),400),pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.25,0,0),300),pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.35,0,0),300)]},
  dad:{description:'Open hand thumb taps forehead',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.35,0,0),400),pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.25,0,0),300)]},
  brother:{description:'L from forehead + index together',poses:[pose(L_SHAPE,w(-0.35,0,0),400),pose(POINT,w(0,0,0),400)]},
  sister:{description:'L from chin + index together',poses:[pose(L_SHAPE,w(-0.2,0,0),400),pose(POINT,w(0,0,0),400)]},
  baby:{description:'Arms cradle and rock',poses:[pose(FLAT,w(0.2,0,0.15),400),pose(FLAT,w(0.2,0,-0.15),400),pose(FLAT,w(0.2,0,0.15),400)]},
  friend:{description:'Hooked index fingers link',poses:[pose(X_SHAPE,w(0,0.2,0),400),pose(X_SHAPE,w(0,-0.2,0),400)]},
  teacher:{description:'O hands at forehead move forward',poses:[pose(O_SHAPE,w(-0.3,0,0),400),pose(O_SHAPE,w(0.1,0,0),400)]},
  student:{description:'Grab from palm, put to head',poses:[pose(OPEN,w(0.2,0,0),400),pose(O_SHAPE,w(-0.3,0,0),400)]},
  doctor:{description:'D taps wrist pulse',poses:[pose(D_SHAPE,w(0.2,0.2,0),400),pose(D_SHAPE,w(0.1,0.1,0),300),pose(D_SHAPE,w(0.2,0.2,0),300)]},
  person:{description:'P hands move down',poses:[pose({thumb:f(0.2,0,0,0.3),index:STRAIGHT,middle:f(0.8,0.5,0.3,-0.1),ring:CURLED,pinky:CURLED},w(0,0,0),400),pose({thumb:f(0.2,0,0,0.3),index:STRAIGHT,middle:f(0.8,0.5,0.3,-0.1),ring:CURLED,pinky:CURLED},w(0.3,0,0),400)]},
  boy:{description:'Flat O at forehead - cap brim grip',poses:[pose(O_SHAPE,w(-0.35,0,0),400),pose(O_SHAPE,w(-0.25,0,0),300)]},
  girl:{description:'Thumb traces jawline',poses:[pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.1,0,0),400)]},
  man:{description:'Open hand from forehead to chest',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.35,0,0),400),pose({...OPEN,thumb:f(0,0,0,0.5)},w(0,0,0),400)]},
  woman:{description:'Open hand from chin to chest',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...OPEN,thumb:f(0,0,0,0.5)},w(0,0,0),400)]},
  // === THINGS & PLACES ===
  food:{description:'Flat O taps mouth',poses:[pose(O_SHAPE,w(-0.2,0,0),400),pose(O_SHAPE,w(-0.1,0,0),300)]},
  water:{description:'W taps chin',poses:[pose(W_SHAPE,w(-0.2,0,0),350),pose(W_SHAPE,w(-0.1,0,0),300),pose(W_SHAPE,w(-0.2,0,0),300)]},
  home:{description:'Flat O from cheek to jaw',poses:[pose(O_SHAPE,w(-0.2,0,0),400),pose(O_SHAPE,w(-0.1,0.2,0),400)]},
  house:{description:'Flat hands form roof then walls',poses:[pose(FLAT,w(0,0,0.15),400),pose(FLAT,w(0.2,0,-0.15),400)]},
  school:{description:'Flat hands clap twice',poses:[pose(FLAT,w(0,0,0),300),pose(FLAT,w(0.15,0,0),250),pose(FLAT,w(0,0,0),300)]},
  car:{description:'Fists steer wheel',poses:[pose(FIST,w(0,0,0.25),400),pose(FIST,w(0,0,-0.25),400)]},
  phone:{description:'Y hand at ear',poses:[pose(Y_SHAPE,w(-0.2,0.1,0),700)]},
  money:{description:'Flat O taps on palm',poses:[pose(O_SHAPE,w(0.2,0,0),350),pose(O_SHAPE,w(0.1,0,0),300),pose(O_SHAPE,w(0.2,0,0),300)]},
  book:{description:'Flat hands open like book',poses:[
    { right: pose(FLAT,w(0,0,-0.15),400), left: pose(FLAT,w(0,0,0.15),400) },
    { right: pose(FLAT,w(0,0,0.2),400),   left: pose(FLAT,w(0,0,-0.2),400) }
  ]},
  door:{description:'B hands open then close',poses:[pose(FLAT,w(0,0,-0.1),400),pose(FLAT,w(0,0,0.3),400)]},
  name:{description:'H taps on H',poses:[pose({...U_SHAPE},w(0,-0.5,0),400),pose({...U_SHAPE},w(0,-0.8,0),400)]},
  store:{description:'Flat O hands swing out',poses:[pose(O_SHAPE,w(0,0,0),400),pose(O_SHAPE,w(0.2,0.2,0),400)]},
  hospital:{description:'H draws cross on arm',poses:[pose({...U_SHAPE},w(0,0.2,0),400),pose({...U_SHAPE},w(0.2,0,0),400)]},
  // === OTHER COMMON WORDS ===
  not:{description:'Thumb under chin flicks forward',poses:[pose({...FIST,thumb:f(0,0,0,0.5)},w(-0.2,0,0),400),pose({...FIST,thumb:f(0,0,0,0.5)},w(0.2,0,0),400)]},
  dont:{description:'Open hands cross then separate',poses:[pose(OPEN,w(0,0,0),350),pose(OPEN,w(0.2,0.3,0),350)]},
  can:{description:'Fists push down together',poses:[pose(FIST,w(0,0,0),400),pose(FIST,w(0.3,0,0),400)]},
  cannot:{description:'Index passes under palm out',poses:[pose(POINT,w(0,0,0),400),pose(POINT,w(0.3,0,0.3),400)]},
  with:{description:'Fists come together',poses:[pose(FIST,w(0,0,0.2),400),pose(FIST,w(0,0,0),400)]},
  without:{description:'Fists together then apart',poses:[pose(FIST,w(0,0,0),400),pose(OPEN,w(0,0,0.3),400)]},
  very:{description:'V hands pull apart',poses:[pose(V_SHAPE,w(0,0,0),400),pose(V_SHAPE,w(0,0,0.3),400)]},
  really:{description:'Index from mouth forward',poses:[pose(POINT,w(-0.2,0,0),400),pose(POINT,w(0.2,0,0),400)]},
  all:{description:'Flat hand circles then drops into palm',poses:[pose(FLAT,w(0,0.2,0),400),pose(FLAT,w(0,-0.2,0),400)]},
  many:{description:'Fists flick open repeatedly',poses:[pose(FIST,w(0,0,0),300),pose(OPEN,w(0,0,0),300),pose(FIST,w(0,0,0),300)]},
  because:{description:'Index at forehead swipes to side',poses:[pose(POINT,w(-0.3,0,0),400),pose(FLAT,w(-0.3,0.3,0),400)]},
  but:{description:'Index fingers cross then pull apart',poses:[pose(POINT,w(0,0,0),400),pose(POINT,w(0,0,0.3),400)]},
  about:{description:'Index circles around flat hand',poses:[pose(POINT,w(0,0.2,0),350),pose(POINT,w(0,-0.2,0),350)]},
  every:{description:'A-thumbs brush down',poses:[pose({...FIST,thumb:f(0,0,0,0.5)},w(0,0,-0.1),400),pose({...FIST,thumb:f(0,0,0,0.5)},w(0.2,0,0.1),400)]},
  // === EXTRA DAILY USE ===
  bathroom:{description:'T hand shakes',poses:[pose({...FIST,thumb:f(0.3,0.2,0,0.1)},w(0,0,0.2),350),pose({...FIST,thumb:f(0.3,0.2,0,0.1)},w(0,0,-0.2),350),pose({...FIST,thumb:f(0.3,0.2,0,0.1)},w(0,0,0.2),300)]},
  perfect:{description:'F hands circle and meet',poses:[pose(F_SHAPE,w(0,0.2,0),400),pose(F_SHAPE,w(0,-0.2,0),400)]},
  wonderful:{description:'Open hands push forward from face',poses:[pose(OPEN,w(-0.2,0,0),400),pose(OPEN,w(0.3,0,0),400)]},
  beautiful2:{description:'Open hand circles face then closes',poses:[pose(OPEN,w(-0.2,0.2,0),400),pose(O_SHAPE,w(-0.1,0,0),400)]},
  awesome:{description:'Claw hands at temples expand',poses:[pose(CLAW,w(-0.3,0,0),400),pose(OPEN,w(0,0,0),400)]},
  nice:{description:'Flat hand slides across palm',poses:[pose(FLAT,w(0,0.1,0),400),pose(FLAT,w(0,-0.1,0),400)]},
  fine:{description:'Open hand thumb on chest',poses:[pose({...OPEN,thumb:f(0,0,0,0.5)},w(-0.1,0,0),700)]},
  cool:{description:'Thumb and pinky near face',poses:[pose(Y_SHAPE,w(-0.2,0,0.1),700)]},
  wow:{description:'Open hands shake in surprise',poses:[pose(OPEN,w(-0.1,0,0.2),350),pose(OPEN,w(-0.1,0,-0.2),350)]},
  color:{description:'Wiggle fingers at chin',poses:[pose(OPEN,w(-0.2,0,0),350),pose(CLAW,w(-0.2,0,0),300),pose(OPEN,w(-0.2,0,0),300)]},
  music:{description:'Flat hand waves over arm',poses:[pose(FLAT,w(0,0,0.2),350),pose(FLAT,w(0,0,-0.2),350),pose(FLAT,w(0,0,0.2),300)]},
  happy2:{description:'Brush up chest twice',poses:[pose(FLAT,w(0,0,0),350),pose(FLAT,w(-0.25,0,0),350)]},
  dance:{description:'V fingers sway on palm',poses:[pose(V_SHAPE,w(0,0,0.2),350),pose(V_SHAPE,w(0,0,-0.2),350),pose(V_SHAPE,w(0,0,0.2),300)]},
  sing:{description:'Flat hand waves forward from mouth',poses:[pose(FLAT,w(-0.2,0,0),400),pose(FLAT,w(0,0.2,0),400)]},
  movie:{description:'Open hand shakes on palm',poses:[pose(OPEN,w(0.2,0,0.15),350),pose(OPEN,w(0.2,0,-0.15),350)]},
  internet:{description:'Middle fingers tap and turn',poses:[pose({...OPEN,middle:SLIGHT},w(0,0,0.15),400),pose({...OPEN,middle:SLIGHT},w(0,0,-0.15),400)]},
  computer:{description:'C hand moves up arm',poses:[pose(C_SHAPE,w(0,0,0),400),pose(C_SHAPE,w(-0.2,0,0),400)]},
  chat:{description:'Flat hands alternate at mouth',poses:[pose(FLAT,w(-0.2,0,0),350),pose(FLAT,w(-0.1,0.1,0),300),pose(FLAT,w(-0.2,0,0),300)]},
  share:{description:'Flat hand sweeps between',poses:[pose(FLAT,w(0,0,0.1),400),pose(FLAT,w(0.2,0,-0.1),400)]},
};

export const REST_POSE = ASL_POSES[' '];
