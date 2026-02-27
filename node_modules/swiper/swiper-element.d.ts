// @ts-ignore
import { Swiper, SwiperOptions } from './types/index.d.ts';

declare const register: () => void;

// prettier-ignore
interface SwiperContainerEventMap extends Omit<HTMLElementEventMap, 'click' | 'progress' | 'keypress' | 'resize' | 'touchstart' | 'touchmove' | 'touchend' | 'transitionend' | 'transitionstart'> {
  /**
   * Event will be fired in when autoplay started
   */
  autoplaystart: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when autoplay stopped
   */
  autoplaystop: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on autoplay pause
   */
  autoplaypause: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on autoplay resume
   */
  autoplayresume: CustomEvent<[swiper: Swiper]>;
  /**
   * Event triggers continuously while autoplay is enabled. It contains time left (in ms) before transition to next slide and percentage of that time related to autoplay delay
   */
  autoplaytimeleft: CustomEvent<[swiper: Swiper, timeLeft: number, percentage: number]>;
  /**
   * Event will be fired when slide changed with autoplay
   */
  autoplay: CustomEvent<[swiper: Swiper]>;/**
   * Event will be fired on window hash change
   */
  hashchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when swiper updates the hash
   */
  hashset: CustomEvent<[swiper: Swiper]>;/**
   * Event will be fired on key press
   */
  keypress: CustomEvent<[swiper: Swiper, keyCode: string]>;/**
   * Event will be fired on navigation hide
   */
  navigationhide: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on navigation show
   */
  navigationshow: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on navigation prev button click
   */
  navigationprev: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on navigation next button click
   */
  navigationnext: CustomEvent<[swiper: Swiper]>;/**
   * Event will be fired on mousewheel scroll
   */
  scroll: CustomEvent<[swiper: Swiper, event: WheelEvent]>;/**
   * Event will be fired after pagination rendered
   */
  paginationrender: CustomEvent<[swiper: Swiper, paginationEl: HTMLElement]>;

  /**
   * Event will be fired when pagination updated
   */
  paginationupdate: CustomEvent<[swiper: Swiper, paginationEl: HTMLElement]>;

  /**
   * Event will be fired on pagination hide
   */
  paginationhide: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired on pagination show
   */
  paginationshow: CustomEvent<[swiper: Swiper]>;/**
   * Event will be fired on draggable scrollbar drag start
   */
  scrollbardragstart: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired on draggable scrollbar drag move
   */
  scrollbardragmove: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired on draggable scrollbar drag end
   */
  scrollbardragend: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;/**
   * Event will be fired on zoom change
   */
  zoomchange: CustomEvent<[swiper: Swiper, scale: number, imageEl: HTMLElement, slideEl: HTMLElement]>;

  
  /**
   * Fired right after Swiper initialization.
   * @note Note that with `swiper.on('init')` syntax it will
   * work only in case you set `init: false` parameter.
   *
   * @example
   * ```js
   * const swiper = new Swiper('.swiper', {
   *   init: CustomEvent<[false,
   *   // other parameters
   * }]>;
   * swiper.on('init', function() {
   *  // do something
   * });
   * // init Swiper
   * swiper.init();
   * ```
   *
   * @example
   * ```js
   * // Otherwise use it as the parameter:
   * const swiper = new Swiper('.swiper', {
   *   // other parameters
   *   on: CustomEvent<[{
   *     init: function  {
   *       // do something
   *     },
   *   }
   * })]>;
   * ```
   */
  init: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired right before Swiper destroyed
   */
  beforedestroy: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired after slides and their sizes are calculated and updated
   */
  slidesupdated: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when currently active slide is changed
   */
  slidechange: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired in the beginning of animation to other slide (next or previous).
   */
  slidechangetransitionstart: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired after animation to other slide (next or previous).
   */
  slidechangetransitionend: CustomEvent<[swiper: Swiper]>;

  /**
   * Same as "slideChangeTransitionStart" but for "forward" direction only
   */
  slidenexttransitionstart: CustomEvent<[swiper: Swiper]>;

  /**
   * Same as "slideChangeTransitionEnd" but for "forward" direction only
   */
  slidenexttransitionend: CustomEvent<[swiper: Swiper]>;

  /**
   * Same as "slideChangeTransitionStart" but for "backward" direction only
   */
  slideprevtransitionstart: CustomEvent<[swiper: Swiper]>;

  /**
   * Same as "slideChangeTransitionEnd" but for "backward" direction only
   */
  slideprevtransitionend: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired in the beginning of transition.
   */
  transitionstart: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired after transition.
   */
  transitionend: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired when user touch Swiper. Receives `pointerdown` event as an arguments.
   */
  touchstart: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user touch and move finger over Swiper. Receives `pointermove` event as an arguments.
   */
  touchmove: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user touch and move finger over Swiper in direction opposite to direction parameter. Receives `pointermove` event as an arguments.
   */
  touchmoveopposite: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user touch and move finger over Swiper and move it. Receives `pointermove` event as an arguments.
   */
  slidermove: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user release Swiper. Receives `pointerup` event as an arguments.
   */
  touchend: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user click/tap on Swiper. Receives `pointerup` event as an arguments.
   */
  click: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user click/tap on Swiper. Receives `pointerup` event as an arguments.
   */
  tap: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when user double tap on Swiper's container. Receives `pointerup` event as an arguments
   */
  doubletap: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;

  /**
   * Event will be fired when Swiper progress is changed, as an arguments it receives progress that is always from 0 to 1
   */
  progress: CustomEvent<[swiper: Swiper, progress: number]>;

  /**
   * Event will be fired when Swiper reach its beginning (initial position)
   */
  reachbeginning: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired when Swiper reach last slide
   */
  reachend: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired when Swiper goes to beginning or end position
   */
  toedge: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired when Swiper goes from beginning or end position
   */
  fromedge: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired when swiper's wrapper change its position. Receives current translate value as an arguments
   */
  settranslate: CustomEvent<[swiper: Swiper, translate: number]>;

  /**
   * Event will be fired everytime when swiper starts animation. Receives current transition duration (in ms) as an arguments
   */
  settransition: CustomEvent<[swiper: Swiper, transition: number]>;

  /**
   * Event will be fired on window resize right before swiper's onresize manipulation
   */
  resize: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired if observer is enabled and it detects DOM mutations
   */
  observerupdate: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired right before "loop fix"
   */
  beforeloopfix: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired after "loop fix"
   */
  loopfix: CustomEvent<[swiper: Swiper]>;

  /**
   * Event will be fired on breakpoint change
   */
  breakpoint: CustomEvent<[swiper: Swiper, breakpointParams: SwiperOptions]>;

  /**
   * !INTERNAL: Event will fired right before breakpoint change
   */
 

  /**
   * !INTERNAL: Event will fired after setting CSS classes on swiper container element
   */
 

  /**
   * !INTERNAL: Event will fired after setting CSS classes on swiper slide element
   */
 

  /**
   * !INTERNAL: Event will fired after setting CSS classes on all swiper slides
   */
 

  /**
   * !INTERNAL: Event will fired as soon as swiper instance available (before init)
   */
 

  /**
   * !INTERNAL: Event will be fired on free mode touch end (release) and there will no be momentum
   */
 

  /**
   * Event will fired on active index change
   */
  activeindexchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired on snap index change
   */
  snapindexchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired on real index change
   */
  realindexchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired right after initialization
   */
  afterinit: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired right before initialization
   */
  beforeinit: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired before resize handler
   */
  beforeresize: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired before slide change transition start
   */
  beforeslidechangestart: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will fired before transition start
   */
  beforetransitionstart: CustomEvent<[swiper: Swiper, speed: number, internal: any]>; // what is internal?
  /**
   * Event will fired on direction change
   */
  changedirection: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when user double click/tap on Swiper
   */
  doubleclick: CustomEvent<[swiper: Swiper, event: MouseEvent | TouchEvent | PointerEvent]>;
  /**
   * Event will be fired on swiper destroy
   */
  destroy: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on momentum bounce
   */
  momentumbounce: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired on orientation change (e.g. landscape -> portrait)
   */
  orientationchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired in the beginning of animation of resetting slide to current one
   */
  slideresettransitionstart: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired in the end of animation of resetting slide to current one
   */
  slideresettransitionend: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired with first touch/drag move
   */
  sliderfirstmove: CustomEvent<[swiper: Swiper, event: TouchEvent]>;
  /**
   * Event will be fired when number of slides has changed
   */
  slideslengthchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when slides grid has changed
   */
  slidesgridlengthchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when snap grid has changed
   */
  snapgridlengthchange: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired after swiper.update() call
   */
  update: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when swiper is locked (when `watchOverflow` enabled)
   */
  lock: CustomEvent<[swiper: Swiper]>;
  /**
   * Event will be fired when swiper is unlocked (when `watchOverflow` enabled)
   */
  unlock: CustomEvent<[swiper: Swiper]>;
  
}

interface SwiperContainer extends HTMLElement {}
interface SwiperContainer extends SwiperOptions {
  swiper: Swiper;
  initialize: () => void;
  injectStyles: string[];
  injectStylesUrls: string[];
  eventsPrefix: string;
  addEventListener<K extends keyof SwiperContainerEventMap>(
    type: K,
    listener: (this: SwiperContainer, ev: SwiperContainerEventMap[K]) => any,
    options?: boolean | AddEventListenerOptions,
  ): void;
  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions,
  ): void;
  removeEventListener<K extends keyof SwiperContainerEventMap>(
    type: K,
    listener: (this: SwiperContainer, ev: SwiperContainerEventMap[K]) => any,
    options?: boolean | EventListenerOptions,
  ): void;
  removeEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | EventListenerOptions,
  ): void;
}

interface SwiperSlide extends HTMLElement {
  lazy: string | boolean;
}

declare global {
  interface HTMLElementTagNameMap {
    'swiper-container': SwiperContainer;
    'swiper-slide': SwiperSlide;
  }
}

export { SwiperContainer, SwiperSlide, register };
