interface RiveOptions {
  locateFile(file: string): string;
}

declare function Rive(options?: RiveOptions): Promise<RiveCanvas>;
export default Rive;

/**
 * RiveCanvas is the main export object that contains references to different Rive classes to help
 * build the Rive render loop for low-level API usage. In addition, this contains multiple methods
 * that help aid in setup, such as loading in a Rive file, creating the renderer, and
 * starting/finishing the render loop (requestAnimationFrame)
 */
export interface RiveCanvas {
  Alignment: AlignmentFactory;
  CanvasRenderer: typeof CanvasRenderer;
  LinearAnimationInstance: typeof LinearAnimationInstance;
  StateMachineInstance: typeof StateMachineInstance;
  CustomFileAssetLoader: typeof CustomFileAssetLoader;
  Mat2D: typeof Mat2D;
  Vec2D: typeof Vec2D;
  AABB: AABB;
  SMIInput: typeof SMIInput;
  renderFactory: CanvasRenderFactory;

  BlendMode: typeof BlendMode;
  FillRule: typeof FillRule;
  Fit: typeof Fit;
  RenderPaintStyle: typeof RenderPaintStyle;
  StrokeCap: typeof StrokeCap;
  StrokeJoin: typeof StrokeJoin;
  decodeAudio: DecodeAudio;
  decodeImage: DecodeImage;
  decodeFont: DecodeFont;

  /**
   * Loads a Rive file for the runtime and returns a Rive-specific File class
   *
   * @param buffer - Array buffer of a Rive file
   * @param assetLoader - FileAssetLoader used to optionally customize loading of font and image assets
   * @param enableRiveAssetCDN - boolean flag to allow loading assets from the Rive CDN, enabled by default.
   * @returns A Promise for a Rive File class
   */
  load(
    buffer: Uint8Array,
    assetLoader?: FileAssetLoader,
    enableRiveAssetCDN?: boolean,
  ): Promise<File>;

  /**
   * Creates the renderer to draw the Rive on the provided canvas element
   *
   * @param canvas - Canvas to draw the Rive on
   * @param useOffscreenRenderer - Option for those using the WebGL-variant of the Rive JS library.
   * This uses an offscreen renderer to draw on the canvas, allowing for multiple Rives/canvases on
   * a given screen. It is highly recommended to set this to `true` when using with the
   * `@rive-app/webgl` package
   * @returns A Rive CanvasRenderer (Canvas2D) or Renderer (WebGL) class
   */
  makeRenderer(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    useOffscreenRenderer?: boolean,
  ): WrappedRenderer;

  /**
   * Computes how the Rive is laid out onto the canvas
   * @param {Fit} fit - Fit enum (i.e Fit.contain)
   * @param alignment - Alignment enum (i.e Alignment.center)
   * @param frame - AABB object representing the bounds of the canvas frame
   * @param content - AABB object representing the bounds of what to draw the Rive onto
   * (i.e an artboard's size)
   * @param scaleFactor - Scale factor of the artboard when using `Fit.layout`
   * @returns Mat2D - A Mat2D view matrix
   */
  computeAlignment(
    fit: Fit,
    alignment: Alignment,
    frame: AABB,
    content: AABB,
    scaleFactor?: number,
  ): Mat2D;

  mapXY(matrix: Mat2D, canvasPoints: Vec2D): Vec2D;
  /**
   * A Rive-specific requestAnimationFrame function; this must be used instead of the global
   * requestAnimationFrame function.
   * @param cb - Callback function to call with an elapsed timestamp
   * @returns number - An ID of the requestAnimationFrame request
   */
  requestAnimationFrame(cb: (timestamp: DOMHighResTimeStamp) => void): number;
  /**
   * A Rive-specific cancelAnimationFrame function; this must be used instead of the global
   * cancelAnimationFrame function.
   * @param requestID - ID of the requestAnimationFrame request to cancel
   */
  cancelAnimationFrame(requestID: number): void;
  /**
   * A Rive-specific function to "flush" queued up draw calls from using the renderer.
   *
   * This should only be invoked once at the end of a loop in a regular JS
   * requestAnimationFrame loop, and should not be used with the Rive-wrapped
   * requestAnimationFrame (aka, the requestAnimationFrame() API on this object) as that
   * API will handle flushing the draw calls implicitly.
   */
  resolveAnimationFrame(): void;
  /**
   * Debugging tool to showcase the FPS in the corner of the screen in a new div. If a callback
   * function is provided, this function passes the FPS count to the callback instead of creating a
   * new div so the client can decide what to do with that data.
   */
  enableFPSCounter(cb?: (fps: number) => void): void;
  /**
   * Debugging tool to remove the FPS counter that displays from enableFPSCounter
   */
  disableFPSCounter(): void;

  /**
   * Cleans up any WASM-generate objects that need to be destroyed manually.
   * This should be called when you wish to remove a rive animation from view.
   */
  cleanup(): void;

  /**
   * Returns whether or not there are Rive Listeners configured on a given StateMachineInstance
   * @param stateMachine - StateMachineInstance to check for Listeners
   * @returns bool - Boolean of if there are Listners on the state machine
   */
  hasListeners(stateMachine: StateMachineInstance): boolean;
}

//////////////
// RENDERER //
//////////////

/**
 * Rive wrapper around a rendering context for a canvas element, implementing a subset of the APIs
 * from the rendering context interface
 */
export declare class RendererWrapper {
  /**
   * Saves the state of the canvas and pushes it onto a stack
   *
   * For the underlying API, check
   * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/save
   */
  save(): void;
  /**
   * Restores the most recent state of the canvas saved on the stack
   *
   * For the underlying API, check
   * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/restore
   */
  restore(): void;
  transform(tranform: Mat2D): void;
  drawPath(path: RenderPath, paint: RenderPaint): void;
  clipPath(path: RenderPath): void;
  /**
   * Calls the context's clearRect() function to clear the entire canvas. Crucial to call
   * this at the start of the render loop to clear the canvas before drawing the next frame
   *
   * For the underlying API, check
   * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/clearRect
   */
  clear(): void;
  delete(): void;
  flush(): void;
  translate(x: number, y: number): void;
  rotate(angle: number): void;
}

export declare class RenderPathWrapper {
  reset(): void;
  addPath(path: CommandPath, transform: Mat2D): void;
  fillRule(value: FillRule): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  cubicTo(
    ox: number,
    oy: number,
    ix: number,
    iy: number,
    x: number,
    y: number,
  ): void;
  close(): void;
}

export declare class RenderPaintWrapper {
  color(value: number): void;
  thickness(value: number): void;
  join(value: StrokeJoin): void;
  cap(value: StrokeCap): void;
  blendMode(value: BlendMode): void;
  style(value: RenderPaintStyle): void;
  linearGradient(sx: number, sy: number, ex: number, ey: number): void;
  radialGradient(sx: number, sy: number, ex: number, ey: number): void;
  addStop(color: number, stop: number): void;
  completeGradient(): void;
}

/**
 * Renderer returned when Rive makes a renderer via `makeRenderer()`
 */
export declare class Renderer extends RendererWrapper {
  /**
   * Aligns the Rive content on the canvas space
   * @param fit - Fit enum value
   * @param alignment - Alignment enum value
   * @param frame - Bounds of the canvas space
   * @param content - Bounds of the Rive content
   * @param _scaleFactor - Scale factor of the artboard when using `Fit.layout`
   */
  align(
    fit: Fit,
    alignment: Alignment,
    frame: AABB,
    content: AABB,
    scaleFactor?: number,
  ): void;
}

export declare class CommandPath {}

export declare class RenderPath extends RenderPathWrapper {}

export declare class RenderPaint extends RenderPaintWrapper {}

/////////////////////
// CANVAS RENDERER //
/////////////////////
export declare class CanvasRenderer extends Renderer {
  constructor(
    ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  );
}

type OmittedCanvasRenderingContext2DMethods =
  | "createConicGradient"
  | "createImageData"
  | "createLinearGradient"
  | "createPattern"
  | "createRadialGradient"
  | "getContextAttributes"
  | "getImageData"
  | "getLineDash"
  | "getTransform"
  | "isContextLost"
  | "isPointInPath"
  | "isPointInStroke"
  | "measureText";

/**
 * Proxy class that handles calls to a CanvasRenderer instance and handles Rive-related rendering calls such
 * as `save`, `restore`, `transform`, and more, effectively overriding and/or wrapping Canvas2D context
 * APIs for Rive-specific purposes. Other calls not intentionally overridden are passed through to the
 * Canvas2D context directly.
 *
 * Note: Currently, any calls to the Canvas2D context that you expect to return a value (i.e. `isPointInStroke()`)
 * will return undefined
 */
export type CanvasRendererProxy = CanvasRenderer &
  Omit<CanvasRenderingContext2D, OmittedCanvasRenderingContext2DMethods>;

/**
 * Renderer type for `makeRenderer()` that returns Renderer (webgl) or a CanvasRendererProxy (canvas2d)
 */
export type WrappedRenderer = Renderer | CanvasRendererProxy;

export declare class CanvasRenderPaint extends RenderPaint {
  draw(
    ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
    path: RenderPath,
  ): void;
}

export declare class CanvasRenderPath extends RenderPath {}

export interface CanvasRenderFactory {
  makeRenderPaint(): CanvasRenderPaint;
  makeRenderPath(): CanvasRenderPath;
}

export class Audio {
  unref(): void;
}
export interface AudioCallback {
  (audio: Audio): void;
}
export interface DecodeAudio {
  (bytes: Uint8Array, callback: AudioCallback): void;
}
export class Image {
  unref(): void;
}
export interface ImageCallback {
  (image: Image): void;
}
export interface DecodeImage {
  (bytes: Uint8Array, callback: ImageCallback): void;
}
export class Font {
  unref(): void;
}
export interface FontCallback {
  (font: Font): void;
}
export interface DecodeFont {
  (bytes: Uint8Array, callback: FontCallback): void;
}

//////////
// File //
//////////
/**
 * Rive-specific File class that provides a number of functions to load instances of Artboards
 */
export declare class File {
  /**
   * Returns the first Artboard found in the Rive file as a new Artboard instance
   * @returns An Artboard instance
   */
  defaultArtboard(): Artboard; // rive::ArtboardInstance
  /**
   * Returns the named Artboard found in the Rive file as a new Artboard instance
   * @param name - Name of the Artboard to create an instance for
   */
  artboardByName(name: string): Artboard; // rive::ArtboardInstance
  /**
   * Returns a new Artboard instance for the Artboard at the given index in the Rive file
   * @param index - Index of the Artboard in the file to create an Artboard instance for
   */
  artboardByIndex(index: number): Artboard; // rive::ArtboardInstance
  /**
   * Returns the number of Artboards in the Rive File
   * @returns Number of artboards in the Rive file
   */
  artboardCount(): number;

  delete(): void;
}

/**
 * Rive class representing an Artboard instance. Use this class to create instances for
 * LinearAnimations, StateMachines, Nodes, Bones, and more. This Artboard instance should also be
 * advanced in the drawing render loop.
 *
 * Important: Make sure to delete this instance when it's no longer in use via the `delete()`
 * method. This deletes the underlying c++ reference and frees up the backing WASM object. This can
 * be done in cases where the user navigates away from the page with this animation, the canvas is
 * unmounted, etc.
 */
export declare class Artboard {
  /**
   * Get the name of this Artboard instance
   */
  get name(): string;
  /**
   * Get the bounds of this Artboard instance
   */
  get bounds(): AABB;
  get hasAudio(): boolean;
  get frameOrigin(): boolean;
  set frameOrigin(val: boolean);
  /**
   * Getter and setter for the artboard volume
   */
  get volume(): number;
  set volume(val: number);

  /**
   * Getter and setter for the artboard width
   */
  get artboardWidth(): number;
  set artboardWidth(val: number);

  /**
   * Getter and setter for the artboard height
   */
  get artboardHeight(): number;
  set artboardHeight(val: number);

  /**
   * Getter and setter used in rendering and canvas/artboard resizing
   */
  get devicePixelRatioUsed(): number;
  set devicePixelRatioUsed(val: number);

  /**
   * Reset the artboard size to its original values
   */
  resetArtboardSize(): void;
  /**
   * Deletes the underlying instance created via the WASM. It's important to clean up this
   * instance when no longer in use
   */
  delete(): void;
  /**
   * Advances the Artboard instance by the set amount of seconds. This method updates each object
   * in the Artboard with any changes that animations apply on properties of the objects. This
   * should be called after calling `advance()` of a LinearAnimationInstance or StateMachineInstance
   * @param sec - Scrub the Artboard instance by a number of seconds
   */
  advance(sec: number): boolean;
  /**
   * Draws the artboard with a given rendering context.
   * @param renderer - Renderer context to draw with
   */
  draw(renderer: CanvasRenderer | Renderer): void;
  /**
   * Creates a LinearAnimation for the animation with the given name
   *
   * Note: This does not create a LinearAnimationInstance to advance in the render loop.
   * That needs to be created separately.
   *
   * @param name - Name of the animation to create a LinearAnimation reference for
   * @returns A new LinearAnimation object
   */
  animationByName(name: string): LinearAnimation;
  /**
   * Creates a LinearAnimation for the animation with the given index
   *
   * Note: This does not create a LinearAnimationInstance to advance in the render loop.
   * That needs to be created separately.
   *
   * @param index - Index of the animation to create a LinearAnimation reference for
   * @returns A new LinearAnimation object
   */
  animationByIndex(index: number): LinearAnimation;
  /**
   * Returns the number of animations in the artboard
   * @returns Number of animations on the Artboard
   */
  animationCount(): number;
  /**
   * Creates a StateMachine for the state machine with the given name.
   *
   * Note: This does not create a StateMachineInstance to advance in the render loop.
   * That needs to be created separately.
   *
   * @param name - Name of the state machine to create a StateMachine reference for
   * @returns A new StateMachine object
   */
  stateMachineByName(name: string): StateMachine;
  /**
   * Creates a StateMachine for the state machine with the given index
   *
   * Note: This does not create a StateMachineInstance to advance in the render loop.
   * That needs to be created separately.
   *
   * @param index - Index of the state machine to create a StateMachine reference for
   * @returns A new StateMachine object
   */
  stateMachineByIndex(index: number): StateMachine;
  /**
   * Returns the number of state machines in the artboard
   * @returns Number of state machines on the Artboard
   */
  stateMachineCount(): number;
  /**
   * Returns a reference for a Bone object of a given name.
   * Learn more: https://rive.app/community/doc/bones/docYyQwxrgI5
   *
   * @param name - Name of the Bone to grab a reference to
   */
  bone(name: string): Bone;
  /**
   * Returns a reference for a Node object of a given name from the Artboard hierarchy
   * @param name - Name of the Node from the Artboard hierarchy to grab a reference to
   */
  node(name: string): Node;
  /**
   * Returns a reference for a root Bone object of a given name
   * @param name - Name of the root Bone to grab a reference to
   */
  rootBone(name: string): RootBone;
  /**
   * Returns a reference for a transform component object of a given name
   * @param name - Name of the transform component to grab a reference to
   */
  transformComponent(name: string): TransformComponent;
  /**
   * Returns a reference for a TextValueRun object to get/set a text value for
   * @param name - Name of the Text Run to grab a reference to
   */
  textRun(name: string): TextValueRun;
  /**
   * Returns a reference for a SMIInput object to get/set an input value for
   * @param name - Name of the Input to grab a reference to
   * @param path - Path of where the input exists at an artboard level
   */
  inputByPath(name: string, path: string): SMIInput;
  /**
   * Returns a reference for a TextValueRun object to get/set a text value for
   * @param name - Name of the Text Run to grab a reference to
   * @param path - Path of where the text exists at an artboard level
   */
  textByPath(name: string, path: string): TextValueRun;
  /**
   * Getter and setter for the artboard width
   */
  get width(): number;
  set width(val: number);
  /**
   * Getter and setter for the artboard height
   */
  get height(): number;
  set height(val: number);
  /**
   * Reset the artboard size to the original value
   */
  resetArtboardSize(): void;
}

export declare class Bone extends TransformComponent {
  /**
   * Length of the bone
   */
  length: number;
}

export declare class RootBone extends Bone {
  /**
   * X coordinate of the position on the RootBone
   */
  x: number;
  /**
   * Y coordinate of the position on the RootBone
   */
  y: number;
}

/**
 * Representation of a node in the Artboard hierarchy (i.e group, shape, etc.)
 */
export declare class Node extends TransformComponent {
  /**
   * X coordinate of the position on the Node
   */
  x: number;
  /**
   * Y coordinate of the position on the RootBone
   */
  y: number;
}

export declare class TransformComponent {
  rotation: number;
  scaleX: number;
  scaleY: number;
  worldTransform(): Mat2D;
  parentWorldTransform(result: Mat2D): void;
}

///////////////
// Animation //
///////////////
/**
 * Rive class representing a LinearAnimation instance. Use this class to advance and control a
 * particular animation in the render loop (i.e speed, scrub, mix, etc.).
 *
 * Important: Make sure to delete this instance when it's no longer in use via the `delete()`
 * method. This deletes the underlying c++ reference and frees up the backing WASM object. This can
 * be done in cases where the user navigates away from the page with this animation, the canvas is
 * unmounted, etc.
 */
export declare class LinearAnimationInstance {
  /**
   * Create a new LinearAnimationInstance reference
   * @param animation - A LinearAnimation reference retrieved via the Artboard
   * (i.e `artboard.animationByName('foo')`)
   * @param artboard - The Artboard instance for this animation
   */
  constructor(animation: LinearAnimation, artboard: Artboard);
  get name(): string;
  get duration(): number;
  get fps(): number;
  get workStart(): number;
  get workEnd(): number;
  get loopValue(): number;
  get speed(): number;
  /**
   * Number of seconds the animation has advanced by
   */
  time: number;
  /**
   * Flag to determine if the animation looped (this is reset when the loop restarts)
   */
  didLoop: boolean;
  /**
   * Advances/scrubs the LinearAnimationInstance by the set amount of seconds. Note that this only
   * moves the "time" in the animation, but does not apply changes to the properties in the
   * Artboard. This must be called before the `apply()` method of LinearAnimationInstance.
   *
   * @param sec - Scrub the animation instance by a number of seconds
   */
  advance(sec: number): boolean;
  /**
   * Apply a mixing value on the animation instance. This is useful if you are looking to blend
   * multiple animations together and want to dictate a strength for each of the animations played
   * back. This also applies new values to properties of objects on the Artboard according to the
   * keys of the animation.
   * This must be called after the `advance()` method of `LinearAnimationInstance`
   *
   * @param mix 0-1 the strength of the animation in the animations mix.
   */
  apply(mix: number): void;
  /**
   * Deletes the underlying instance created via the WASM. It's important to clean up this instance
   * when no longer in use
   */
  delete(): void;
}

export declare class TextValueRun {
  /**
   * Getter for the name of the Text Run
   */
  get name(): string;
  /**
   * Getter for text value of the Text Run
   */
  get text(): string;
  /**
   * Setter for the text value of the Text Run
   */
  set text(val: string);
}

/**
 * Rive Event interface for "General" custom events defined in the Rive editor. Each event has a
 * name and optionally some other custom properties and a type
 */
export interface RiveEvent {
  /**
   * Name of the event fired
   */
  name: string;
  /**
   * Optional type of the specific kind of event fired (i.e. General, OpenUrl)
   */
  type?: number;
  /**
   * Optional custom properties defined on the event
   */
  properties?: RiveEventCustomProperties;
  /**
   * Optional elapsed time since the event specifically occurred
   */
  delay?: number;
}

/**
 * A specific Rive Event type for "OpenUrl" events. This event type has a URL and optionally a
 * target property to dictate how to open the URL
 */
export interface OpenUrlEvent extends RiveEvent {
  /**
   * URL to open when the event is invoked
   */
  url: string;
  /**
   * Where to display the linked URL
   */
  target?: string;
}

/**
 * A Rive Event may have any number of optional custom properties defined on itself with variable names
 * and values that are either a number, boolean, or string
 */
export interface RiveEventCustomProperties {
  /**
   * Custom property may be named anything in the Rive editor, and given a value of
   * a number, boolean, or string type
   */
  [key: string]: number | boolean | string;
}

export declare class LinearAnimation {
  /**
   * The animation's loop type
   */
  get loopValue(): number;
  /**
   * Name of the LinearAnimation
   */
  get name(): string;
}

export declare class StateMachine {
  /**
   * Name of the StateMachine
   */
  get name(): string;
}

/**
 * Rive class representing a StateMachine instance. Use this class to advance and control a
 * particular state machine in the render loop (i.e scrub, grab state machine inputs, set up
 * listener events, etc.).
 *
 * Important: Make sure to delete this instance when it's no longer in use via the `delete()`
 * method. This deletes the underlying c++ reference and frees up the backing WASM object. This can
 * be done in cases where the user navigates away from the page with this animation, the canvas is
 * unmounted, etc.
 */
export declare class StateMachineInstance {
  /**
   * Create a new StateMachineInstance reference
   * @param stateMachine - A StateMachine retrieved via the Artboard
   * (i.e `artboard.stateMachineByName('foo')`)
   * @param artboard - The Artboard instance for this state machine
   */
  constructor(stateMachine: StateMachine, artboard: Artboard);
  get name(): string;
  /**
   * Returns the number of inputs associated with this state machine
   * @returns Number of inputs
   */
  inputCount(): number;
  /**
   * Returns the state machine input at the given index
   * @param i - Index to retrieve the state machine input at
   * @returns SMIInput reference
   */
  input(i: number): SMIInput;
  /**
   * Advances/scrubs the StateMachineInstance by the set amount of seconds. Note that this does not
   * apply changes to the properties of objects in the Artboard yet.
   * @param sec - Scrub the state machine instance by a number of seconds
   */
  advance(sec: number): boolean;
  /**
   * Advances/scrubs the StateMachineInstance by the set amount of seconds. Note that this will
   * apply changes to the properties of objects in the Artboard.
   * @param sec - Scrub the state machine instance by a number of seconds
   */
  advanceAndApply(sec: number): boolean;
  /**
   * Returns the number of states changed while the state machine played
   * @returns Number of states changed in the duration of the state machine played
   */
  stateChangedCount(): number;
  /**
   * Returns the name of the state/animation transitioned to, given the index in the array of state
   * changes in total
   * @param i
   * @returns Name of the state/animation transitioned to
   */
  stateChangedNameByIndex(i: number): string;

  /**
   * Returns the number of events reported from the last advance call
   * @returns Number of events reported
   */
  reportedEventCount(): number;

  /**
   * Returns a RiveEvent object emitted from the last advance call at the given index
   * of a list of potentially multiple events. If an event at the index is not found,
   * undefined is returned.
   * @param i index of the event reported in a list of potentially multiple events
   * @returns RiveEvent or extended RiveEvent object returned, or undefined
   */
  reportedEventAt(i: number): OpenUrlEvent | RiveEvent | undefined;

  /**
   * Notifies the state machine that the pointer has pressed down at the given coordinate in
   * Artboard space. Internally, Rive may advance a state machine if the listener coordinate is of
   * interest at a given moment.
   *
   * @param x - X coordinate
   * @param y - Y coordinate
   */
  pointerDown(x: number, y: number): void;
  /**
   * Notifies the state machine that the pointer has moved to the given coordinate in
   * Artboard space. Internally, Rive may advance a state machine if the listener coordinate is of
   * interest at a given moment.
   *
   * @param x - X coordinate
   * @param y - Y coordinate
   */
  pointerMove(x: number, y: number): void;
  /**
   * Notifies the state machine that the pointer has released at the given coordinate in
   * Artboard space. Internally, Rive may advance a state machine if the listener coordinate is of
   * interest at a given moment.
   * @param x - X coordinate
   * @param y - Y coordinate
   */
  pointerUp(x: number, y: number): void;

  /**
   * Deletes the underlying instance created via the WASM. It's important to clean up this instance
   * when no longer in use
   */
  delete(): void;
}

export declare class SMIInput {
  // TODO: Keep only the base SMIInput properties and make SMIBool, SMINumber, SMITriger extend it
  static bool: number;
  static number: number;
  static trigger: number;

  /**
   * Getter for name of the state machine input
   */
  get name(): string;
  get type(): number;
  /**
   * Getter for value of the state machine input
   */
  get value(): boolean | number | undefined;
  /**
   * Setter for value of the state machine input
   */
  set value(val: boolean | number | undefined);
  /**
   * Fires a trigger input on a state machine
   */
  fire(): void;
  asBool(): SMIInput;
  asNumber(): SMIInput;
  asTrigger(): SMIInput;
}

export declare class SMIBool {}

export declare class SMINumber {}

export declare class SMITrigger {}

///////////
// ENUMS //
///////////

export enum Fit {
  fill,
  contain,
  cover,
  fitWidth,
  fitHeight,
  none,
  scaleDown,
  layout,
}

export enum RenderPaintStyle {
  fill,
  stroke,
}

export enum FillRule {
  nonZero,
  evenOdd,
}

export enum StrokeCap {
  butt,
  round,
  square,
}
export enum StrokeJoin {
  miter,
  round,
  bevel,
}

export enum BlendMode {
  srcOver = 3,
  screen = 14,
  overlay = 15,
  darken = 16,
  lighten = 17,
  colorDodge = 18,
  colorBurn = 19,
  hardLight = 20,
  softLight = 21,
  difference = 22,
  exclusion = 23,
  multiply = 24,
  hue = 25,
  saturation = 26,
  color = 27,
  luminosity = 28,
}

///////////
// UTILS //
///////////

export declare class Alignment {
  get x(): number;
  get y(): number;
}

export declare class AlignmentFactory {
  get topLeft(): Alignment;
  get topCenter(): Alignment;
  get topRight(): Alignment;
  get centerLeft(): Alignment;
  get center(): Alignment;
  get centerRight(): Alignment;
  get bottomLeft(): Alignment;
  get bottomCenter(): Alignment;
  get bottomRight(): Alignment;
}

/**
 * Axis-aligned bounding box
 */
export interface AABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/**
 * Column-major matrix described by the following:
 *  | xx  yx  tx |
 *  | xy  yy  ty |
 *  |  0   0   1 |
 */
export declare class Mat2D {
  xx: number;
  xy: number;
  yx: number;
  yy: number;
  tx: number;
  ty: number;
  /**
   * Returns whether or not a matrix could be inverted, and if yes, sets the resulting Mat2D into
   * the passed-in `mat` parameter
   *
   * @param mat - Reference Mat2D to store the newly inverted matrix into if successful
   * @returns True if the matrix could be inverted
   */
  invert(mat: Mat2D): boolean;

  /**
   * Deletes the underlying CPP object created for this instance
   */
  delete(): void;
}

/**
 * Rive Vector class
 */
export declare class Vec2D {
  constructor(x: number, y: number);
  /**
   * Returns the x coordinate of the vector
   */
  x(): number;
  /**
   * Returns the y coordinate of the vector
   */
  y(): number;

  /**
   * Deletes the underlying CPP object created for this instance
   */
  delete(): void;
}

/**
 * Rive class representing a FileAsset with relevant metadata fields to describe
 * an asset associated wtih the Rive File
 */
export declare class FileAsset {
  name: string;
  fileExtension: string;
  uniqueFilename: string;
  isAudio: boolean;
  isImage: boolean;
  isFont: boolean;
  cdnUuid: string;

  decode(bytes: Uint8Array): void;
}

/**
 * Rive class extending the FileAsset that exposes a `setAudioSource()` API with a
 * decoded Audio (via the `decodeAudio()` API) to set a new Audio on the Rive FileAsset
 */
export declare class AudioAsset extends FileAsset {
  setAudioSource(audio: Audio): void;
}

/**
 * Rive class extending the FileAsset that exposes a `setRenderImage()` API with a
 * decoded Image (via the `decodeImage()` API) to set a new Image on the Rive FileAsset
 */
export declare class ImageAsset extends FileAsset {
  setRenderImage(image: Image): void;
}

/**
 * Rive class extending the FileAsset that exposes a `setFont()` API with a
 * decoded Font (via the `decodeFont()` API) to set a new Font on the Rive FileAsset
 */
export declare class FontAsset extends FileAsset {
  setFont(font: Font): void;
}

export declare class FileAssetLoader {}

export declare class CustomFileAssetLoader extends FileAssetLoader {
  constructor({ loadContents }: { loadContents: Function });
  loadContents(asset: FileAsset, bytes: any): boolean;
}
