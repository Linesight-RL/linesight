//| Sciter.d.ts v0.24.2
//| https://github.com/MustafaHi/sciter-vscode

interface Behaviors
{
    frame: frame;
    "frame-set": {state: array};
    history: history;
    
    from: form;
    select: select;
    calender: calender
    textarea: textarea;
    edit: edit;
    masked: masked;
    
    plaintext: plaintext;
    richtext: richtext;
    
    vlist: virtualList;
    scrollbar: scrollbar;
    
    lottie: lottie;
    video: video;
}

interface frame
{
    /** Initiates loading of the document from the URL.  
      * calls `newdocument/complete` events */
    loadFile(path: string): boolean;
    /** Initiates loading of the document from the html string or bytes.  
      * calls `newdocument/complete` events */
    loadHtml(html: string|ArrayBuffer, baseURL?: string): boolean;
    /** Clear the content of the frame by loading empty document in it. */
    loadEmpty(): void;
    /** Save document to the file in UTF-8 encoding. */
    saveFile(path: string): boolean;
    /** Save document into ArrayBuffer as sequence of UTF-8 encoded bytes. */
    saveBytes(): ArrayBuffer;
    
    readonly document: Document;
    /**  Get/Set key/value map of media variables used by the document. */
    mediaVars: object;
    /** URL of document loaded into the frame. */
    url: string;
}

interface history
{
    /** Goes back in navigational stack, returns true if navigation was successful. */
    back(): boolean;
    /** Goes forward in navigational stack, returns true if navigation was successful. */
    forward(): boolean;

    /** Depth of history in backward direction. */
    readonly length: number;
    /** Depth of history in forward direction. */
    readonly forwardLength: number;
}

interface form
{
    /** Submits content of the form if its action attribute is defined on the form. */
    submit(): void;
    /** Resets input elements to their initial values. */
    reset(): void;
}

interface select
{
    /** Shows popup list of options */
    showPopup(): void;
    /** Closes popup list of options if it is open */
    hidePopup(): void;

    /** Reference to DOM element that holds `<option>` list. */
    options: Element;
}

interface virtualList
{
    navigateTo(target: number|"start"|"end"|"pagenext"|"pageprior"|"itemnext"|"itemprior"): void;
    /** scroll to given record number. By default it performs animated scroll.  
        Returns DOM element representing the record. */
    advanceTo(target: number): Element;

    /** Get of first visible item in the buffer. */
    readonly firstVisibleItem: Element;
    /** Get of last visible item in the buffer. */
    readonly lastVisibleItem: Element;
    readonly firstBufferIndex: number;
    readonly lastBufferIndex: number;
    readonly itemsTotal: number;
    itemsBefore: number;
    itemsAfter: number;
}

interface textarea
{
    /** Index of first selected character. */
    readonly selectionStart: number;
    /** Index of last select character+1/ */
    readonly selectionEnd: number;
    /** selected text or empty sting if there is no selection or selection is collapsed. */
    readonly selectionText: string;

    selectAll(): void;
    selectRange(start: number, end: number): void;
    appendText(text: string): boolean;
    /** Replace select text with given text. */
    insertText(text: string): boolean;
    /** Remove selected text. */
    removeText(): boolean;
}

interface scrollbar
{
    /** Sets values of scrollbar element - position, min, max,
     * page - reflects to size of scrollbar slider,
     * step - increment value of on arrow buttons clicks. */
    values(position:number, min:number, max:number, page:number, step:number): void;

    readonly max: number;
    readonly min: number;
    /** Page value, size of scrollbar's slider. */
    readonly page: number;
    /** Defines position increment/decrement of clicks on arrow buttons. */
    readonly step: number;

    /** Current slider position. */
    position: number;
}

interface calender
{
    /** Gets/Sets current view mode. */
    mode: "days"|"months"|"years"|"century";
    /** Decrements the value by 1 or a specified number.
     * Depends of current mode it will advance either day or month or year or decade. */
    stepDown(steps?: number): void;
    /** Increments the value by 1 or by a specified number.
     * Depends of current mode it will advance either day or month or year or decade. */
    stepUp(steps?: number): void;
}

interface edit
{
    readonly selectionStart, readonly selectionEnd: number;
    /** Returns selected text or empty string if there is no selection. */
    readonly selectionText: string;

    /** Select whole content. */
    selectAll(): void;
    /** selects text between start (included)
     * and end (excluded) position. If start and end are omitted - removes selection. */
    selectRange(start?: number, end?: number): void;
    /** Remove selected text. */
    removeText(): void;
    /** Insert text at caret position, if selection is not empty removes selected text before insertion. */
    insertText(text: string): void;
    /** Appends the text at the end of existing text. */
    appendText(text: string): void;
}

interface masked
{
    /** This property allows to define structure of masked input "manually" and with more control.
     * Mask definition is an array of strings (rendered as static separators) and objects.
     * Each object defines editable regions and may have following fields: */
    mask: string|array;
    type: "integer"|"text"|"enum";
    /** Defines length of the region in characters */
    width: number;
    /** Defines CSS class of generated span element */
    class: string;
    min, max, step: number;
    /** If true then this `type:integer` field is prefixed by zeros */
    "leading-zero": boolean;
    /** Defines list of enumerable cases for `type:enum`
     * this region allows to input only those predefined cases. */
    items: enum;
    /** Defines filter of allowed characters in this `type:text` field. */
    filter: string;
}

/** Provides playback of Lottie animations. It parses Adobe After Effects
 * animations exported as json with Bodymovin and renders them natively inside the Sciter.  
 * [Documentation](https://gitlab.com/sciter-engine/sciter-js-sdk/-/blob/main/docs/md/behaviors/behavior-lottie.md) */
interface lottie
{
    /** Reports playback status. If true animation is playing at the moment. */
    readonly playing: boolean;
    /** float, speed multiplier, 1.0 by default. */
    speed: number;
    loop: boolean;
    /** Current frame in range [0..frames). */
    frame: number;
    /** Total number of frames in animation. */
    readonly frames: number;
    /** Current animation position, number in range 0.0 .. 1.0 */
    position: number;
    /** Total duration of full animation loop as defined by animation designer. */
    readonly duration: number;
    /** Array (list) of marker definitions.
     * Each definition is an array (tuple): [tagName:string, startFrame: integer, endFrame: integer]. */
    readonly markers: array;

    /** Load (potentially asynchronously) animation from JSON file at URL. */
    load(url: string): boolean;
    /** Start playback. If first/last frame is provided will play only frames in the range. */
    play(firstFrame?: number, lastFrame: number): boolean;
    /** Stop (pause) animation. */
    stop(): boolean;
    /** Update animation properties dynamically at runtime. */
    update(keyPath: string, propName: string, value: Color|number): boolean;
}

interface plaintext
{
    /** Get/Set text line at given index */
    children: string[];
    /** String content, lines seperated by \r\n */
    content: string|string[];
    readonly lines: number;
    readonly selectionStart: [lineNumber: number, linePosition: number];
    readonly selectionEnd  : [lineNumber: number, linePosition: number];
    readonly selectionText: string;

    /** Load Content from URL */
    load(url: string): boolean;
    /** Save Content to URL(file path) */
    save(url: string): boolean;
    /** Select text range; */
    selectRange(startLine: number, startPosition: number, endLine: number, endPosition: number): void;
    /** Select all text; */
    selectAll(): boolean;
    /** Append line/s at the end of the text; */
    appendLine(text: string|string[]): boolean;
    /** Inserts line/s at line index; */
    insertLine(index: number, text: string|string[]): boolean;
    /** Remove line/s starting from index */
    removeLine(index: number, count?: number): boolean;
    /** Performs transactional (undoable) content update.  
     * [Documentation](https://gitlab.com/sciter-engine/sciter-js-sdk/-/blob/main/docs/md/behaviors/behavior-richtext.md#richtextupdatemutatorfunctiontctx-bool) */
    update(mutator: (context: Transaction) => {}): boolean;
}

interface richtext
{
    /** Get/Set url of loaded document. */
    url: string;

    /** Load Content from URL */
    load(url: string): boolean;
    /** loads content from bytes or string (html source) into the editor;
     * url is used to resolve relative URLs (if any). */
    load(html: string|ArrayBuffer, url?: string): boolean;
    /** Save Content to URL(file path) */
    save(url: string): boolean;
    /** Clear the content by loading empty document in it. */
    loadEmpty(): boolean;
    /** Set content to the html at given selection. */
    sourceToContent(html: string, url: string, selectionStart: number, selectionEnd: number): boolean;
    /** Return content and selection as an array. */
    contentToSource(): [html: string, url: string, selectionStart: number, selectionEnd: number];
    /** Performs transactional (undoable) content update.  
     * [Documentation](https://gitlab.com/sciter-engine/sciter-js-sdk/-/blob/main/docs/md/behaviors/behavior-richtext.md#richtextupdatemutatorfunctiontctx-bool) */
    update(mutator: (context: Transaction) => {}): boolean;
}

interface Transaction
{
    /** Add or change value of one attribute. */
    setAttribute(el: Element, name: string, value: string): void;
    /** Remove one attribute. */
    removeAttribute(el: Element, name: string): void;
    /** Change tag of the element. */
    setTag(el: Element, name: string): void;
    /** Change node text. */
    setText(node: Node, text: string): void;
    /** splits node at offset position until the parent element.
     * Similar to pressing ENTER in the middle of paragraph -
     * text node and p[aragraph] element will be split to two paragraphs; */
    split(node: Node, offset: number, parentEl: Element): [node: Node, offset: number];
    /** Wrap the range into element. */
    wrap(startNode: Node, startOffset: number, endNote: Node, endOffset: number, tag: string): void;
    /** Remove the element and add it content to parent element. */
    unwrap(el: Element): void;
    /** Same as `Element.execCommand()` but all mutations will go into this transaction. */
    execCommand(command: string, params?: object|string): boolean;

    /** Insert text at given node/offset position. */
    insertText(at: Node|number, text: string): [node: Node, offset: number];
    /** Insert HTML at given node/offset position, return list of nodes inserted; */
    insertHTML(at: Node|number, html: string): Node[];
    /** Insert node at given node/offset position. */
    insertNode(at: Node|number, html: string): [node: Node, offset: number];
    
    /** Delete current selected range (if any). */
    deleteSelection(): [node: Node, offset: number];
    deleteRange(startNode: Node, startOffset: number, endNote: Node, endOffset: number): void;
    /** Delete given node or element. */
    deleteNode(node: Node): void;
}

interface video
{
    /** Report playback status. If true then video is playing at the moment. */
    readonly isPlaying: boolean;
    /** If video playback has reached the end of the movie. */
    readonly isEnded: boolean;
    /** Duration in seconds of the movie. If duration is not available it returns 0. */
    readonly duration: number;
    /** Reports natural width and height of video frame of the movie. */
    readonly height, readonly width: number;
    /** Reports video box rectangle in pixels relative to the content box of the element.
     * Note if sizing attribute is "cover" then either x or y can be negative. */
    readonly renderingBox: [x, y, width, height];
    /** float (0.0...1.0). Current volume level of audio track.
     * 1.0 correspond to 0db, 0.0 (mute) -100db. */
    audioVolume: number;
    /** float ( -1.0 ... +1.0 ). Current stereo balance. */
    audioBalance: number;
    /** float, Get/Set current playback position, in seconds. */
    position: number;

    /** Loads video file into the player. use `play()` to start. */
    load(url: string): boolean;
    /** Stops video playback and unloads the movie. */
    unload(): void;
    /** Start playback at current `position` */
    play(): void;
    stop(): void;
}

interface Document extends Element {
    /** Return image associated with provided arbitrary url, or assign one if image is provided too.  
     * This method also allow you to use the image in CSS by it URL.
     */
    bindImage(url: string, image?: Image): Image;

    /** Returns absolute path of provided relative path using the document URL as a base. */
    url(relpath ?: string): string;

    /** Subscribe to any DOM event */
    on(event: keyof typeof eventType, selector?: string, handler: eventFunction): void;
    on(event: keyof typeof domEvent, handler: eventFunction): void;

    /* NATIVE */

    body: Element;
    head: Element;
    /** Root(html) element */
    documentElement: Element;
    /** document loading state - `complete` | `interactive` */
    readyState:  'complete' | 'interactive';
    createElement(tag: string): Element;
    createTextNode(): Node;
    createComment(): Comment;
    createDocumentFragment();
    createNodeIterator(root: string, whatToShow?: string, filter?: string): NodeIterator;
}
declare var Document: {
    new(): Document;
};

declare var document: Document;

enum domEvent {
    parsed,
    ready,
    DOMContentLoaded,
    complete,
    close,
    unload,
    beforeunload,
    closerequest
}

interface Element extends Node, Behaviors {
    /** Get element matching the css selector */
    $(query: string): Element;
    /** Get array of elements matching the css selector */
    $$(query: string): array<Element>;
    /** Select parent element that match the query */
    $p(query: string): Element;
    /** Owner element selector, useful to get owner of menu */
    $o(query: string): Element;
    /** Check element match the selector */
    $is(query: string): boolean;
    /** Posts a function or event to event queue. */
    post(eventOrHandler: function(this: Element, ...any) | Event, avoidDuplicates?: boolean): boolean;
    /** Fire event asynchronously, `Event.target` will be set to this element,  
     * use `dispatchEvent` for sync method
     * @return `false` if event is canceled with `Event.preventDefault()`.
     */
    postEvent(event: Event, avoidDuplicates?: boolean): boolean;
    /** jQuery style event subscription:  
        @param event `^name` for handling events in capturing phase
        @param query subscribe to all children that match the css selector otherwise this element
        @param handler `Function(Event, Element)` - `this` is set to the element the handler is attached to
    */
    on(event: keyof typeof eventType, query: string, handler: eventFunction): Element;
    on(event: keyof typeof eventType, handler: eventFunction): Element;
    off(eventOrHandler: keyof typeof eventType|string|Function): Element;
    /** jQuery style event subscription to application wide events:  
     *  The element gets unsubscribed automatically when it is disconnected from DOM
        @param event `^name` for handling events in capturing phase
        @param handler `Function(Event, Element)` - `this` is set to the element the handler is attached to
    */
    onGlobalEvent(event: string, handler: function(this: Element, ...any)): Element;
    /** Unsubscribe this element from particular event, if no argument is provided unsubscribe from all events */
    offGlobalEvent(eventOrHandler?: string | function(this: Element, ...any)): Element;
    /** Starts timer on element.
     *  If the element already has a timer with the same callback, it first gets removed and timer is restarted.
     *  This allows to implement effective throttling (debounce).
     *  @param callback `this` is set to the element, `return true` to repeat. */
    timer(milliseconds: number, callback: function(this: Element, ...any): void|boolean): boolean;
    /** Removes content of the element, makes it empty. */
    clear(): boolean;
    /** Interaction with native behaviors attached to the element. */
    xcall(name: string, ...args): any
    /** Removes the element and moves its content in place in the DOM. */
    unwrapElement(): boolean;
    /** Wraps range of nodes from start to end into wrap element - opposite action to `unwrapElement()` */
    wrapNodes(start: Node, end: Node, wrap: Element);
    /** Reports state and allowance of particular command. The method accepts the same parameters as the `Element.execCommand()`.  */
    checkCommand(command: string, params?: object|string): 1|2;
    /** Execute undoable behavior specific commands. */
    execCommand(command: string, params?: object|string): boolean;
    /** Immediate mode drawing "ports".
     *  Functions assigned to these properties will be called when the element is rendered on screen
     *  so they can draw anything on top (or below) of default HTML/CSS rendering. */
    paintBackground: function(Graphics);
    /** Immediate mode drawing "ports".
     *  Functions assigned to these properties will be called when the element is rendered on screen
     *  so they can draw anything on top (or below) of default HTML/CSS rendering. */
    paintForeground: function(Graphics);
    /** Immediate mode drawing "ports".
     *  Functions assigned to these properties will be called when the element is rendered on screen
     *  so they can draw anything on top (or below) of default HTML/CSS rendering. */
    paintContent: function(Graphics);
    /** Immediate mode drawing "ports".
     *  Functions assigned to these properties will be called when the element is rendered on screen
     *  so they can draw anything on top (or below) of default HTML/CSS rendering. */
    paintOutline: function(Graphics);
    /** Schedules re-paint of the element. This will trigger `Element.paintXXXX` calls. */
    requestPaint(): void;
    /** Force repaint immediately */
    flushPaint(): void;

    /** Shows the popup element or VNode (JSX) in out-of-canvas popup window on desktop. */
    popup(popup: Element | VNode, params?: popupParams): void;
    /** Show this element as out-of-canvas popup window on desktop. 
     * @param referencePoint `1-9`, see keyboard numpad for the meaning.
    */
    popupAt(x: number, y: number, referencePoint?: number): void;
    /** The method offers "manual" animation support.  
     *  `function(progress: 0.0...1.0)`: true | false  
     *  Sciter will call handler with animation frame rate passing current progress value.
     *  return false to stop animation. */
    animate(handler: Function, params: animateParams): void;
    /** Make the element "airborn" - to be replaced outside of host window */
    takeOff(params: takeoffParams): void;
    /** Append element as last child */
    append(JSX: JSX): void;
    /** Insert element as the first child */
    prepend(JSX: JSX): void;
    /** Replace content by element */
    content(JSX: JSX): void;
    /** patches content of the element by JSX using rules of React[or].  
     *  If second parameter is true the function patches only children but not element itself. */
    patch(JSX: JSX, onlyChildren?: true): void;
    /** Patch properties and enqueue rendering */
    componentUpdate(object?: object = null): void;
    /** Return collapsed range (caret position) at point x/y.
     *  x/a are local coordinates - relative to origin of element's inner box. */
    rangeFromPoint(x: number, y: number): Range | null;
    toString(): string;

    /* NATIVE */

    /** Get element matching the css selector */
    querySelector(query: string): Element?;
    /** Get array of elements matching the css selector */
    querySelectorAll(query: string): Element[];
    getElementById(id: string): Element?;
    getElementsByClassName(className: string): Element[];
    getElementsByTagName(tag: string): Element[];
    getElementsByName(name: string): Element[];
    /** Find the closest parent element matching the query selector */
    closest(query: string): Element | null;
    /** Check element match the selector */
    matches(query: string): boolean;
    firstElementChild: Element;
    lastElementChild: Element;
    nextElementSibling: Element;
    previousElementSibling: Element;
    childElementCount: number;
    children: Element[];
    childElement(index: number): Element;
    readonly ownerDocument: Document;

    appendChild(node: Node);
    removeChild(node: Node);
    insertBefore(node: Node, refNode: Node);
    insertAfter(node: Node, refNode: Node);
    replaceChild(newNode: Node, refNode: Node);
    insertAdjacentHTML(where: InsertPosition, html: string): void;
    swapWith(element: Element);
    
    style: Style;
    /** Runtime flags and state on element.  
     * Most of Element.State reflect so called CSS pseudo-classes (flags): 
     * `element:visited { color: red; }`
    */
    state: State;
    /** Represents current selection on elements that supports selection:  
       ` <htmlarea>` - WYSIWYG HTML editor;  
        `<plaintext>` - Plain text multiline editor;  
        any other element with `[selectable]` attribute set;   */
    selection: Selection;

    disabled: boolean;
    readonly: boolean;
    checked: boolean;
    src: string;

    readonly attributes: any[string];
    hasAttribute(name: string): boolean;
    getAttribute(name: string): string;
    getAttributeNames(): string[];
    setAttribute(name: string, value: string|number|undefined): void;
    removeAttribute(name: string): void;
    attributes: string[]|number[];
    classList: {
        add(...name: string[]): void;
        remove(...name: string[]): void;
        toggle(name: string, state?: boolean): boolean;
        contains(name: string): boolean;
        length: number;
        readonly entries(): string[];
    }
    /** Returns a drawing context of the canvas, instance of Graphics object. */
    getContext(type: '2d'): Graphics;

    id: string;
    name: string;
    tagName: string;
    tag: string;
    className: string;
    elementIndex: number;
    innerHTML: string;
    outerHTML: string;
    innerText: string;
    value: any;

    scrollBy(x: number, y: number): void;
    scrollBy(options: {
        left?: number;
        top?: number;
        behavior?: "instant" | "smooth";
    }): void;
    scrollTo(x: number, y: number): void;
    scrollTo(options: {
        left?: number;
        top?: number;
        behavior?: "instant" | "smooth";
    }): void;
    scrollIntoView(toTop?: true): void;
    scrollIntoView(options: {
        block?: "start" | "nearest";
        behavior?: "instant" | "smooth";
    }): void;
    readonly clientLeft: number;
    readonly clientTop : number;
    readonly clientWidth: number;
    readonly clientHeight: number;
    readonly scrollLeft: number;
    readonly scrollTop : number;
    readonly scrollRight: number;
    readonly scrollWidth: number;
    readonly scrollHeight: number;
    getBoundingClientRect(): DOMRect;

    click(): void;
    focus(): void;
    /** Call handler each time the event is fired */
    addEventListener(name: string, handler: eventFunction, flags?: object): void;
    removeEventListener(name: string, handler: Function): void;
    /** Fire event synchronously, `Event.target` will be set to this element,  
     * use `postEvent` for async method
     * @return `false` if event is canceled with `Event.preventDefault()`.
     */
    dispatchEvent(event: Event, avoidDuplicates?: boolean): boolean;

    // EventTarget
    ready(event: Event, element: Element): void;
    onclick(event: Event, element: Element): void;
    onchange(event: Event, element: Element): void;
    onkeydown(event: Event, element: Element): void;
    onwheel(event: Event, element: Element): void;
}
declare var Element: {
    new(): Element;
}

type InsertPosition = "beforebegin" | "afterbegin" | "beforeend" | "afterend";
interface popupParams {
    /** 1..9, reference point on anchor border box (see keyboard numpad for the meaning) */
    anchorAt?: number;
    /** 1..9, reference point on popup's margin box. */
    popupAt?: number;
    x?: number;
    y?: number;
}
interface animateParams {
    duration?: number,
    ease?: "linear" | "ease" | "ease-in" | "ease-in-out" | "ease-out" | "quad-in" | "quad-out" | "quad-in-out" | "cubic-in" | "cubic-out" | "cubic-in-out" |  "quart-in" | "quart-out" | "quart-in-out" | "quint-in" | "quint-out" | "quint-in-out" | "sine-in" | "sine-out" | "sine-in-out" |  "expo-in" | "expo-out" | "expo-in-out" | "circ-in" | "circ-out" | "circ-in-out" | "elastic-in" | "elastic-out" | "elastic-in-out" |  "back-in" | "back-out" | "back-in-out" | "x-back-in" | "x-back-out" | "x-back-in-out" | "xx-back-in" | "xx-back-out" | "xx-back-in-out" |  "bounce-in" | "bounce-out" | "bounce-in-out";
    effect?: "blend" | "blend-atop" | "slide-top" | "slide-bottom" | "slide-left" | "slide-right" | "slide-over-top" | "slide-over-bottom" | "slide-over-left" | "slide-over-right" | "remove-top" | "remove-bottom" | "remove-left" | "remove-right" | "scroll-top" | "scroll-bottom" | "scroll-left" | "scroll-right";
    /** Times per second the function is called */
    FPS?: number;
}
interface takeoffParams {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    relativeTo?: "screen" | "document" | "window" | "parent" | "self";
    window?: "attached" | "detached" | "popup";
}

interface DOMRect {
    readonly bottom: number;
    readonly height: number;
    readonly left: number;
    readonly right: number;
    readonly top: number;
    readonly width: number;
    readonly x: number;
    readonly y: number;
}
declare var DOMRect: {
    new(x?: number, y?: number, width?: number, height?: number): DOMRect;
    fromRect(other?: DOMRect): DOMRect;
};

interface Selection
{
    /** `true` if selection is collapsed to one position (anchor === focus) */
    readonly isCollapsed: boolean;
    /** Nearest container element that encloses as anchor as focus positions */
    readonly commonAncestorContainer: Element;
    readonly anchorNode: Node;
    readonly anchorOffset: number;
    /** Caret position */
    readonly focusNode: Node;
    readonly focusOffset: number;
    readonly rangeCount: number;
    readonly type: "Caret" | "Selection" | "Element" | "TableCells";

    /** Collapse selection to current focus (caret) position. */
    collapse(): void;
    /** Collapse selection to anchor or focus (the last in the DOM). */
    collapseToEnd(): void;
    /** Collapse selection to anchor or focus (the first in the DOM). */
    collapseToStart(): void;
    /** `true` if the selection contains the node. */
    containsNode(node: Node): boolean;
    /** Remove selection (but not its content). */
    empty(): void;
    /** Set focus (caret) position without changing anchor position. */
    extend(node: Node, offset: number): void;
    getRangeAt(index: number): Range;
    selectNodeContent(node: Node): void;
    setBaseAndExtent(anchorNode: Node, anchorOffset: number, focusNode: Node, focusOffset: number): void;
    /** Return selected text. */
    toString(): string;
}

/** Runtime flags and state on element.  
 * Most of Element.State reflect so called CSS pseudo-classes (flags): 
 * `element:visited { color: red; }`*/
interface State
{
    /** Computes current min and max widths of the element content. */
    contentWidth(): [minWidth: number, maxWidth: number];
    /** Computes current height of the element content with it given width. */
    contentHeight(width: number): number;
    /** Set/remove mouse capture(forward mouse event)  
     *  `true` - captures mouse events by the element and its sub elements.  
     *  `false` - remove capture if the element owns capture now.  
     *  `"strict"` - mouse events will be delivered to the element only. */
    capture(state: boolean|"strict"): boolean;
    /** Report geometry of the window.  
     *  @param property value type to return.
     *  @param metric value in relation to.
     *  @param relativeTo offset x/y are relative to.
     *  @param asPpx return coordinates in screen pixels otherwise DIPs.
     */
    box(property: "xywh"|"rect"|"position"|"dimension", metric: keyof typeof boxMetric, relativeTo?: keyof typeof boxRelativeTo, asPpx?: boolean): number[];
    box(property: keyof typeof boxProperties, metric: keyof typeof boxMetric, relativeTo?: keyof typeof boxRelativeTo, asPpx?: boolean): number;
    /** Parses length string as CSS length units or percentage and then converts them to CSS pixels.
     *  Perecentage values are computed against element dimensions (inner box). */
    pixelsIn(length: string, orientation?: "horizontal" |  "vertical"): number | undefined;
    /** Maps local element coordinates to window coordinates.
     *  This method accounts affine 2D transformation the element and its parents may have. */
    mapLocalToWindow(x: number, y: number): [x: number, y: number];
    /** Maps point on window to local coordinates of particular element.
     *  This method accounts affine 2D transformation the element and its parents may have. */
    mapWindowToLocal(x: number, y: number): [x: number, y: number];

    focus: boolean;
    readonly ownsfocus: boolean;
    link: boolean;
    visited: boolean;
    hover: boolean;
    selected: boolean;
    current: boolean;
    checked: boolean;
    disabled: boolean;
    readonly: boolean;
    expanded: boolean;
    collapsed: boolean;
    invalid: boolean;
    animating: boolean;
    focusable: boolean;
    anchor: boolean;
    popup: boolean;
    ownspopup: boolean;
    tabfocus: boolean;
    empty: boolean;
    busy: boolean;
    dragover: boolean;
    dragsource: boolean;
    droptarget: boolean;
    moving: boolean;
    copying: boolean;
    pressed: boolean;
    ready: boolean;
    active: boolean;
    /** `False` will prevent reconciliation of element's content by Reactor */
    reactive: boolean;
    /** Runtime value of native behavior attached to the element. Actual for input elements. */
    value: any;
    /** Reports visibility status of the element,  
     *  if `0` then the element is visible in full, otherwise combination of these flags:  
     *  `0x1` - left side of border box is clipped out (invisible).  
     *  `0x2` - top side is clipped.  
     *  `0x4` - right side is clipped.  
     *  `0x8` - bottom side is clipped.  
     *  `0xf` - the element is completely clipped out - invisible. */
    occluded: number;

    /** `True` if this is a root document of the window */
    readonly windowroot: boolean;
    /** Layout manager used by the element at the moment. */
    readonly flow: "default" | "vertical" | "horizontal" | "horizontal-wrap" | "vertical-wrap" | "grid" | "table" | "table-fixed" | "table-row" | "table-body" | "columns" | "stack" | "text" | "null" | "image" | "svg" | "svg-child" | "";
    readonly visible: boolean;
}
enum boxProperties { "xywh", "rect", "position", "dimension", "left", "right", "top", "bottom", "width", "height" }
enum boxMetric { "inner", "border", "padding", "margin", "client", "caret", "icon" }
enum boxRelativeTo { "element", "screen", "window", "document", "parent", "container", "self" }

interface Style {
    getPropertyValue(name: string): string;
    setProperty(name: string, value: string|length, important?: boolean): void;
    removeProperty(name: string): void;
    colorOf(name: string): Color | null;
    pixelsOf(name: string): number | null;
    imageOf(name: string): Image | null;
    /** Get/Set CSS variables applied to the element
     * @return `{name: value...}`
     */
    variables(variables?: object): object;
    setCursor(cursor: Image|null, x: number, y: number): void;


    behavior: string;
    aspect: string;
    prototype: string;
    size: string;
    flow: string;
    fontRenderingMode: "sub-pixel" | "snap-pixel";
    imageRendering: "auto" | "inherit" | "default" | "crispy-edges" | "pixelated" | "optimize-quality" | "optimize-speed";
    contextMenu: string;
    hitMargin: string;
    content: string;
    scrollManner: string;
    verticalScrollbar: string;
    horizontalScrollbar: string;
    textOverflow: string;
    popupPosition: string;


    font: string;
    fontSize: length;
    height: length;
    width: length;

    color: string;
    background: string;
    backgroundColor: string;
    backgroundImage: string;
    foreground: string;
    foregroundColor: string;
    foregroundImage: string;

    [name: string]: string|length;
}

/** An event which takes place in the DOM. */
interface Event {
    /** True if event goes through its target's ancestors in reverse tree order, and false otherwise. */
    readonly bubbles: boolean;
    cancelBubble: boolean;
    /** Can be canceled by invoking the preventDefault() method. */
    readonly cancelable: boolean;
    /** True if event invokes listeners past a ShadowRoot node that is the root of its target, and false otherwise. */
    readonly composed: boolean;
    /** Returns the Element whose event listener's callback is currently being invoked. */
    readonly currentTarget: Element | null;
    /** Returns true if preventDefault() was invoked successfully to indicate cancelation, and false otherwise. */
    readonly defaultPrevented: boolean;
    /** Returns the event's phase, which is one of `NONE`, `CAPTURING_PHASE`, `AT_TARGET`, and `BUBBLING_PHASE`. */
    readonly eventPhase: "NONE"|"CAPTURING_PHASE"|"AT_TARGET"|"BUBBLING_PHASE";
    /** Returns true if event was dispatched by the user agent, and false otherwise. */
    readonly isTrusted: boolean;
    readonly srcElement: Element | null;
    /** The element to which event is dispatched (its target). */
    readonly target: Element | null;
    /** The secondary element which is lossing or gaining focus from/to `target` */
    readonly relatedTarget: Element | null;
    /** Type of event, e.g. "click", "hashchange", or "submit". */
    readonly type: string;
    /** If invoked when the cancelable attribute value is true,
     * and while executing a listener for the event with passive set to false,
     * signals to the operation that caused event to be dispatched that it needs to be canceled. */
    preventDefault(): void;
    /** Invoking this method prevents event from reaching any registered event listeners
     * after the current one finishes running and, when dispatched in a tree,
     * also prevents event from reaching any other objects. */
    stopImmediatePropagation(): void;
    /** When dispatched in a tree, invoking this method prevents event
     * from reaching any objects other than the current object. */
    stopPropagation(): void;
    /** String representation of keyCode "KeyA", "F1", "Enter"... */
    readonly code: string;
    /** keyCode list at [include/sciter-x-key-codes.h](https://gitlab.com/sciter-engine/sciter-js-sdk/-/blob/main/include/sciter-x-key-codes.h) */
    readonly keyCode: number;
    /** Platform's native key code, e.g, wParam in WM_KEYDOWN on Windows. */
    readonly platformKeyCode: string;
    readonly AT_TARGET: number;
    readonly BUBBLING_PHASE: number;
    readonly CAPTURING_PHASE: number;
    readonly NONE: number;

    data, details: any;

    readonly altKey: boolean;
    readonly ctrlKey: boolean;
    /** `command` key on OSX, `win` on Windows */
    readonly metaKey: boolean;
    readonly shiftKey: boolean;
    readonly button: number;
    readonly buttons: number;

    readonly clientX: number;
    readonly clientY: number;
    readonly screenX: number;
    readonly screenY: number;
    readonly windowX: number;
    readonly windowY: number;
    readonly deltaX: number;
    readonly deltaY: number;
    /** `0` - `deltaX/Y` are pixels coming from touch devices,  
     *  `1` - `deltaX/Y` are in "lines" (a.k.a. mouse wheel "ticks"). */
    readonly deltaMode: number;

    /** Coordinates relative to `currentTarget` - the element this event handler is attached to. */
    readonly x: number;
    /** Coordinates relative to `currentTarget` - the element this event handler is attached to. */
    readonly y: number;
    /** Used in some events to indicate auxiliary "source" element. */
    readonly source: Element;
    /** Mouse event is on `foreground-image`, return Element containing the image */
    readonly isOnIcon: Element;

    /** Returns pressed status of the key. */
    keyState(key: string): boolean;
}
declare var Event: {
    new(type: string, options?: EventOptions): Event;
    readonly AT_TARGET: number;
    readonly BUBBLING_PHASE: number;
    readonly CAPTURING_PHASE: number;
    readonly NONE: number;
};
interface EventOptions {
    /** True if event goes through its target's ancestors in reverse tree order, and false otherwise. */
    bubbles?: boolean;
    /** Can be canceled by invoking the preventDefault() method. */
    cancelable?: boolean;
    /** True if event invokes listeners past a ShadowRoot node that is the root of its target, and false otherwise. */
    composed?: boolean;
    /** Property passed to event listener. `evt.data/details` */
    data?,details?: any;
}
type eventFunction = function(Event, Element): void;
enum eventType {
    ready,
    complete,

    click,
    input,
    change,
    press,
    changing,
    submit,
    reset,
    expand,
    collapse,
    statechange,
    visualstatechange,
    disabledstatechange,
    readonlystatechange,
    contextmenu,
    contextmenusetup,
    animationend,
    animationstart,
    animationloop,
    transitionend,
    transitionstart,
    mediachange,
    contentchange,
    inputlangchange,
    pastehtml,
    pastetext,
    pasteimage,
    popuprequest,
    popupready,
    popupdismissing,
    popupdismissed,
    tooltiprequest,

    focus,
    focusin,
    focusout,
    blue,

    mouseMove,
    mouseLeave,
    mouseIdle,
    mousetick,
    mousedown,
    mouseup,
    mousewheel,
    mousedragrequest,
    dblclick,
    doubleclick,
    tripleclick,

    keydown,
    keyup,
    keypress,
    compostionstart,
    compositionend,

    scroll,
    scrollanimationstart,
    scrollanimationend,

    sizechange,
    visibilitychange,

    load,
    error,
    
    drag,
    dragenter,
    dragleave,
    drop,
    dragaccept,
    dropcancel,
    willacceptdrop,

    play,
    ended,
    videocoordinate,
    videoframeready,
}

/** Call function after x time
 * @return Timeout ID for `clearTimeout(ID)`
 */
declare function setTimeout(cb: Function, milliseconds: number): number;
/** Cancel `setTimeout` function by it returned ID */
declare function clearTimeout(tID: number): void;
/** Call function every x amount of time
 * @return Interval ID for `clearInterval(ID)`
 */
declare function setInterval(cb: Function, milliseconds: number): number;
/** Cancel `setInterval` function by it returned ID */
declare function clearInterval(iID: number): void;
/** Call function on every frame
 * @return function ID for `cancelAnimationFrame(ID)`
 */
declare function requestAnimationFrame(cb: Function): number;
/** Cancel `requestAnimationFrame` function by it returned ID */
declare function cancelAnimationFrame(aID: number): void;

declare var console:
{
   log(...arg: any): void;
   warn(...arg: any): void;
   error(...arg: any): void;
}

declare function getComputedStyle(el: Element, pseudoElement?: Element): Style;

/**
 * Format arguments using [C-style printf conventions](https://en.cppreference.com/w/cpp/io/c/fprintf).  
 * Sciter specific:  
    `%v` - print as JSON.stringify(arg);  
    `%V` - print as JSON.stringify(arg, null, " ");
 */
declare function printf(...args: string[]): string;

/**
 * Format arguments using [C-style scanf conventions](https://en.cppreference.com/w/c/io/fscanf).  
 */
declare function scanf(...args: string[]): array<string | number>;

/** 
 * "module" version of stock `eval()` function.  
 * It evaluates the text as a module body. If the url is provided it is used as a base URL
 * for resolving relative paths in `import ... from "relpath"` statements inside.
 * @return module's exported data as an object.
 */
declare function evalModule(text: string, url?: string): any;

/** Loads and executes JavaScript at url synchronously. */
declare function loadScript(url: string): void;

/** Loads and executes JavaScript module at url synchronously. Returns modules exports object */
declare function loadScriptModule(url: string): any;

/** Number of physical screen pixels in logical CSS px (dip) */
declare var devicePixelRatio: float;

/** Current document directory */
declare const __DIR__: string;

declare var globalThis: object;
declare var window: typeof globalThis;


declare function fetch(url: string | Request, params?: fetchParams): Promise<Response>;

interface fetchParams
{
   method?: 'POST'|'GET'|'PUT'|'DELETE';
   mode?: 'cors'|'no-cors'|'same-origin';
   cache?: 'default'|'no-cache'|'reload'|'force-cache'|'only-if-cached';
   credentials?: 'same-origin'|'include'|'omit';
   redirect?: 'follow'|'manual'|'error';
   referrerPolicy?: 'non-referrer-when-downgrade'|'non-referrer'|'origin'|'origin-when-cross-origin'|'same-origin'|'strict-origin'|'strict-origin-when-cross-origin'|'unsafe-url';
   integrity?: string;
   keepalive?: boolean;
   sync?: boolean;
   body?: string|FormData;
   headers?: {
      Accept?: string;
      'Accept-Language'?: string;
      'Content-Type'?: 'application/json'|'application/x-www-form-urlencoded'|'text/plain'|'multipart/form-data';
      'Content-Language'?: string;
      [name: string]: string|boolean;
   }
   /** Callback function to be called on download progress.  
    * Note: total argument can be zero if server does not provide `Content-Length` info. */
   downloadProgress?: (fetched: number, total: number) => void;
}

interface Response
{
   readonly body: string;
   readonly bodyUsed: boolean;
   readonly headers: any;
   readonly ok: boolean;
   readonly redirected: boolean;
   readonly status: number;
   readonly statusText: string;
   readonly type: string;
   readonly url: string;
   /** if true then the request was aborted by `request.abort()` call.
    * @version 5.0.1.1+ */
   readonly aborted: boolean;
   readonly request: Request;
   
   arrayBuffer(): Promise<ArrayBuffer>;
   blob(): Promise<ArrayBuffer>;
   clone(): Response;
   error(): Response;
   redirect(url: string, status?: number): Response;
   formData(): Promise<FormData>;
   json(): Promise<any>;
   text(): Promise<string>;
}

interface Request
{
   cache: "no-cache" | "reload" | "default";
   context: "html" | "image" | "style" | "cursor" | "script" | "data" | "font" | "audio";
   headers: any;
   method: 'POST'|'GET'|'PUT'|'DELETE';
   url: string;
   /** Try to abort current request; Response of aborted request will have `response.aborted` property set to true.
    * @version 5.0.1.1+ */
   abort(): void;
   progress?: (bytesLoaded: number, totalBytes: number) => void;

   /** Appends a new value to existing key inside the object, or adds the key if it does not already exist.  
    * To overwrite existing key/value use `set()`.
    */
   append(key: string, value: any, filename?: string): void;
   /** Sets a new value for an existing key inside the object, or adds the key/value if it does not already exist. */
   set(key: string, value: any, filename?: string): void;
   delete(key: string): void;
   /** Returns an iterator allowing to go through all key/value pairs contained in this object */
   entries(): [key: string, value: any][];
   /** Returns the first value associated with a given key */
   get(key: string): any;
   /** Returns all the values associated with a given key */
   getAll(key: string): any[];
   /** Returns a boolean stating whether object contains a certain key. */
   has(key: string): boolean;
   /** Returns an iterator allowing to go through all keys contained in this object.  */
   keys(): string[];
   /** Returns an iterator allowing to go through all values contained in this object. */
   values(): any[];
}
declare var Request:
{
   new(): Request;
}
interface FormData extends Request
{

}
declare var FormData:
{
   new(): FormData;
}

interface URL
{
   /** `#hash` */
   readonly hash: string;
   /** `sub.domain.org` */
   readonly host: string;
   readonly hostname: string;
   /** Full URL */
   readonly href: string;
   /** `https://sub.domain.org` */
   readonly origin: string;
   /** `/path/without/host` */
   readonly pathname: string;
   readonly port: number;
   /** Protocol type: `https:|http:|file:` */
   readonly protocol: string;
   /** Query paramters: `?q=1&w=w` */
   readonly search: string;

   readonly filename: string;
   readonly dir: string;
   readonly extension: string;

   guessMimeType(): string;
}
declare var URL: {
   new(url: string): URL;
   /** Decode and remove prefix */
   toPath(path: string): string;
   /** Encode and prefix path with `file://` */
   fromPath(path: string): string;
}

/** Creates binary JSON pack/unpack context. */
interface BJSON
{
   /** Serializes JSON data to the ArrayBuffer */
   pack(data: object): ArrayBuffer;
   /** Restore data from BJSON blob
    * @param data previously packed JSON data
    * @param cb function taking `(data)` as argument
   */
   unpack(data: ArrayBuffer, cb: Function): void;
}
declare var BJSON: {
   new(): BJSON;
}

declare var Clipboard:
{
   read(): clipboardObject;
   readText(): string;
   write(data: clipboardObject): boolean;
   writeText(text: string): boolean;
   has(type: "text"|"html"|"image"|"file"|"json"|"link"): boolean;
}
interface clipboardObject
{
   text?: string;
   html?: string;
   json?: any;
   /** List of files path */
   file?: string[];
   link?: { caption: string, url: string };
   image?: Image;
}

/** The Zip class allows access to the content of a zip file or blob.
 * @version 5.0.0.2+
 */
interface Zip
{
   /** Number of files(items) in the Zip. */
   readonly length: number;
   /** Fetch file by it index. */
   item(index: number): ZipItem;
   /** Fetch file by it path (local to the zip). */
   item(path: string ): ZipItem;
}
declare var Zip:
{
   openFile(path: string): Zip;
   openData(data: ArrayBuffer): Zip;
}
interface ZipItem
{
   readonly isDir: boolean;
   readonly isFile: boolean;
   /** Local path of the item inside the zip. */
   readonly path: string;
   /** Data of the item as ArrayBuffer. */
   readonly data: ArrayBuffer;
}

declare var Graphics: {
    new(): Graphics;
    Brush: Brush;
    Color: Color;
    Image: Image;
    Path: Path;
    Text: gText;
};

interface Graphics
{
    lineCap: 'butt'|'round'|'square';
    lineJoin: 'round'|'bevel'|'miter';
    strokeStyle: Color | string | Image;
    lineWidth: number;
    strokeWidth: number;
    fillStyle: Color | string | Image;
    font: string;
    /** @version 5.0.0.5+ */
    canvas: CanvasElement;

    clearRect(x: number, y: number, w: number, h: number): void;
    beginPath(): void;
    moveTo(x: number, y: number): void;
    lineTo(x: number, y: number): void;
    quadraticCurveTo(cpx: number, cpy: number, x: number, y: number): void;
    bezierCurveTo(cp1x: number, cp1y: number, cp2x: number, cp2y: number, x: number, y: number): void;
    arc(x: number, y: number, radius: number, startAngle: number, endAngle: number, antiClockWise?: boolean): void;
    arcTo(x: number, y: number, x2: number, y2: number, radius: number): void;
    ellipse(x: number, y: number, radiusX: number, radiusY: number, rotation: number, startAngle: number, endAngle: number, antiClockWise?: boolean): void;
    rect(x: number, y: number, w: number, h: number): void;
    closePath(): void;
    fill(...args): void;
    fillRect(x: number, y: number, w: number, h: number): void;
    fillText(text: string, x: number, y: number, maxWidth: number): void;
    stroke(...args): void;
    strokeRect(x: number, y: number, w: number, h: number): void;
    setLineDash(...args): void;
    save(): void;
    restore(): void;
    scale(x: number, y: number): void;
    translate(x: number, y: number): void;
    rotate(radian: number, x?: number, y?: number): void;
    transform(a: number, b: number, c: number, d: number, e: number, f: number): void;
    setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void;

    draw(path: Path, params: drawPathParams);
    draw(image: Image, params: drawImageParams);
    draw(text: gText, params: drawTextParams);

    pushLayer(x: number, y: number, w: number, h: number, opacity?: number, filter?: string): void;
    pushLayer(clipAreaName: keyof typeof clipAreaName, opacity?: number, filter?: string): void;
    pushLayer(path: Path, opacity?: number): void;
    pushLayer(mask: Image, useAlpha: boolean, opacity?: number): void;
    popLayer(): void;

    createTile(image: Image): Brush;
    createSolid(color: Color): Brush;
}

interface CanvasElement extends Element {
    height: number;
    width: number;
}

interface drawPathParams {
    x: number;
    y: number;
    fill?: "evenodd" | "nonzero";
    stroke?: boolean;
}

interface drawImageParams {
    x: number;
    y: number;
    width?: number;
    height?: number;
    srcX?: number;
    srcY?: number;
    srcWidth?: number;
    srcHeight?: number;
    opacity?: number;
}

interface drawTextParams {
    x: number;
    y: number;
    /** 1..9, defines meaning of x/y coordinates, see NUMPAD.  
     * `5` - center of text,  
     * `7` - left/top corner, etc. */
    alignment: number;
    fill?: Color;
}

type clipAreaName = "background-area" | "border-box" | "padding-box" | "margin-box" | "context-box";

interface Brush
{
    type: number;
    
    addColorStop(pos: number, color: Color): Brush;
    /** Creates linear gradient brush along the line from x1/y1 to x2/y2 */
    createLinearGradient(x1: number, y1: number, x2: number, y2: number): Brush;
    /** Creates radial gradient brush with center at x/y and radius r */
    createRadialGradient(x: number, y: number, r: number): Brush;
}

interface Color
{
    new(color: string): Color;
    /** float(0..1.0), red channel. */
    readonly r: number;
    /** float(0..1.0), green channel */
    readonly g: number;
    /** float(0..1.0), blue channel. */
    readonly b: number;
    /** float(0..1.0), alpha channel, 0.0 - fully transparent, 1.0 - fully opaque. */
    readonly a: number;
    /** int(0..255), red channel. */
    readonly R: number;
    /** int(0..255), green channel. */
    readonly G: number;
    /** int(0..255), blue channel. */
    readonly B: number;
    /** int(0..255), alpha channel, 0.0 - fully transparent, 1.0 - fully opaque. */
    readonly A: number;
    /** [hue:0..360, saturation:0..1, value: 0..1, alpha: 0..1], HSV color representation. */
    readonly hsv: number[];
    /**  [hue:0..360, saturation:0..1, lightness: 0..1], HSL color representation. */
    readonly hsl: number[];

    /** Produces strings in formats  
     *  `#RRGGBB`, `#RRGGBBAA`, `rgb(255,255,255)` or `rgba(255,255,255,1.0)` */
    toString(type?: "RGB" | "RGBA" | "rgb" | "rgba"): string;
    /** Color packaged to uint32 as `(a << 24) | (b << 16) | (g << 8) | (r)` */
    valueOf(): number;
}
declare var Color: {
    /** Creates `Graphics.Color` instance from r,g,b,a components in float numbers  
     * in `0.0-1.0` range. */
    rgb(r: number, g: number, b: number, a?: number): Color;
    /** Creates `Graphics.Color` instance from r,g,b,a components in integers  
     * in `0-255` range. */
    RGB(r: number, g: number, b: number, a?: number): Color;
    /** Creates `Graphics.Color` instance from HSV components in float numbers  
     * in `0.0-1.0` range but `h` is in `0.0-360.0` range. */
    hsv(h: number, s: number, v: number, a?: number): Color;
    /** Creates `Graphics.Color` instance from HSL components in float numbers  
     * in `0.0-1.0` range but `h` is in `0.0-360.0` range. */
    hsl(r: number, g: number, b: number, a?: number): Color;
}

interface Image
{
    /** Render DOM element onto bitmap. */
    new(width: number, height: number, element: Element): Image;
    /** Render arbitrary graphics on bitmap */
    new(width: number, height: number, canvas: (ctx: Graphics) => void, initColor?: number): Image;

    readonly src: string;
    readonly width: number;
    readonly height: number;
    /** Image format `png`, `webp`, etc. */
    readonly packaging: string;

    /** Static-Method load image from URL return promise of an Image object */
    load(url: string): Promise<Image>;
    /** Draw on the image surface. Image must be a bitmap. */
    update(...arg): void;
    toBytes(format: "png"|"jpeg"|"webp"|"bgra", compression?: number): ArrayBuffer;
    fromBytes(data: ArrayBuffer): Image;
    /** Returns pixel color at x/y. */
    colorAt(x: number, y: number): Color;
    /** Compose this image with src image. */
    compose(src: Image, operation: keyof typeof composeOps, dstX?: number, dstY?: number, srcX?: number, srcY?: number, srcW?: number, srcH?: number): Image;
    /** Return fragment of an image at position.
     * @version 5+
     */
    crop(x: number, y: number, width: number, height: number): Image;
}

enum composeOps {
    "src-over", "dst-over", "src-in", "dst-in", "src-out", "dst-out", "src-atop", "dst-atop", "xor", "copy"
}

interface Path
{
    /** Constructs new path object. accepts SVG's <path>s [d attribute](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d) value. */
    new(svgPath?: string): Path;

    moveTo(x: number, y: number): void;
    lineTo(x: number, y: number): void;
    quadraticCurveTo(cpx: number, cpy: number, x: number, y: number): void;
    bezierCurveTo(cp1x: number, cp1y: number, cp2x: number, cp2y: number, x: number, y: number): void;
    arc(x: number, y: number, radius: number, startAngle: number, endAngle: number, antiClockWise?: boolean): void;
    arcTo(x: number, y: number, x2: number, y2: number, radius: number): void;
    ellipse(x: number, y: number, radiusX: number, radiusY: number, rotation: number, startAngle: number, endAngle: number, antiClockWise?: boolean): void;
    rect(x: number, y: number, w: number, h: number): void;
    closePath(): void;
    isPointInside(x: number, y: number): boolean;
    bounds(): [x1: number, y1: number, x2: number, y2: number];
    /**  */
    combine(type: "union"|"intersect"|"xor"|"exclude", otherPath: Path): Path;
}

interface gText
{
    new(...args: string[]): gText;
    /** CSS style rules to decorate the text including fonts, alignment, borders and background.*/
    style: string;
    readonly lines: number;
    /** Text to render. */
    chars: string;
    class: string;
    
    /** Reports minimal, maximal and used width of the text block. */
    width(): [minWidth: number, maxWidth: number, usedWidth: number];
    /** Sets used width of the text block. Note: `text.lines` property may change after that */
    width(usedWidth: number): void;
    /** Reports content and used height of the text block. */
    height(): [contentHeight: number, usedHeight: number];
    /** Sets used height of the text block. Note: `vertical-align` of text style may change location of glyphs on screen. */
    height(usedHeight: number): void;
    lineMetrics(lineNo: number): [posY: number, height: number, baselineOffset: number];
    /** Textual content of the line */
    lineChars(lineNo: number): string;
}

/** Enable JSX React support */
declare var React: any;

declare namespace JSX {
    // Custom Element documentation support
    interface IntrinsicElements {
        /** Root document element */
        html: any;
        [tagName: string]: any;
    }
}

declare module "@debug" {
    /** Gets call stack item at level */
    export function callStackAt(level: number): callStack;
    /** Catch unhandled exceptions. pass `0` to get info about current function.
     * @param cb function taking (Error) as argument
     */
    export function setUnhandledExeceptionHandler(cb: Function): void;
    /** Redirect console output. make sure to reset `console.log`...
     * @param cb function taking `(subsystem: number, severity: number, msg: any)` as argument
     */
    export function setConsoleOutputHandler(cb: Function): void;
    export function setBreakpointHandler(cb: Function): void;
    export function setBreakpoints(cb: Function): void;
    export function getElementByUID(id: number): Element;
    export function getUIDofElement(el: Element): number;
    export function highlightElement(el: Element): void;
    export function getStyleRulesOfElement(el: Element): Style;
    export function containerId(): number;
    export function objectKind(object: Object): string;
    export function sublimatedValue(value: any, expanded: any): any;
    export function sublimatedValueElements(): any;
    export function frameVariables(id: number): any;
}

interface callStack {
    /** Is that call stack frame is of native function. */
    isNative: boolean;
    functionName: string;
    /** line number of function declaration */
    functionLineNo: number;
    fileName: string;
    /** line number inside the function. */
    LineNo: number;
}

declare module "@env" {
    /** OS identification name: `"Windows-10.1"`... */
    export const OS: string;
    /** OS/platform generic name: `"Windows", "OSX", "Linux", "Android"`, etc. */
    export const PLATFORM: string;
    /** Device type */
    export const DEVICE: "desktop" | "mobile";
    /** Returns two-letter language abbreviation of user's default language. */
    export function language(): string;
    /** Returns two-letter country abbreviation of the user's country */
    export function country(): string;
    export function userName(): string;
    /** Machine network name. */
    export function machineName(): string;
    /** Machine network domain. */
    export function domainName(): string;
    /**
     * Launch file/URL with default system application
     */
    export function launch(path:string): void;
    /** Converts relative path to absolute path using location of `sciter.dll` as a base. */
    export function home(relpath ?: string): string;
    /** Converts relative path to absolute path prefixed by `file://` using location of sciter.dll as a base. */
    export function homeURL(relpath ?: string): string;
    /**
     * Return path of default system folder ie. documents|downloads
     * @param name of default system folder
     * @param relpath relative path to name
     */
    export function path(name: keyof typeof systemPath, relpath ?: string): string;
    /**
     * Return path of default system folder ie. documents|downloads, prefixed by `file://`
     * @param name of default system folder
     * @param relpath relative path to name
     */
    export function pathURL(name: keyof typeof systemPath): string;
    /** Get/Set environment variable */
    export function variable(key: string, value?: string): string;
    /**
     * Execute comma seperated arguments
     */
    export function exec(...args: string[]): void;
}

enum systemPath { "home", "root", "desktop", "applications", "downloads", "documents", "music", "videos", "pictures", "USER_APPDATA" }

declare module "@sciter" {
    export const VERSION: string;
    export const REVISION: string;
    export const QUICKJS_VERSION: string;
    /** Returns first matched DOM element in current document. */
    export function $(query: string): Element;
    /** Returns list (array) of matched DOM elements. */
    export function $$(query: string): array<Element>;
//    export function import(path: string): object;
    /**
     * Load native Sciter extension
     * @param name path to library without .dll/.dylib (relative to sciter.dll)
     */
    export function loadLibrary(name: string): any;
    /** Parses string by "JSON++ rules" returning it actual value: Date, Array, Angle, Hex... */
    export function parseValue(val:string): any;
    /** Converts length to device (screen) pixels */
    export function devicePixels(length: number | string, axis: "width" | "height")
    /** Generate unique id */
    export function uuid(): string;
    /** Subscribe to any DOM event */
    export function on(event: keyof typeof eventType, selector?: string, handler: eventFunction): void;
    /** Unsubscribe to any DOM event */
    export function off(eventOrHandler: keyof typeof eventType | function): void;
    /** Encodes text to sequence of bytes (ArrayBuffer). Default encoding is "utf-8". */
    export function encode(text: string, encoding ?: string): ArrayBuffer;
    /** Decodes sequence of bytes of buffer to string. Default encoding is "utf-8". */
    export function decode(bytes: ArrayBuffer, encoding ?: string): string;
    export function compress(input: ArrayBuffer, method?: "gz" | "gzip" | "lzf"): ArrayBuffer;
    export function decompress(input: ArrayBuffer, method?: "gz" | "gzip" | "lzf"): ArrayBuffer;
    export function toBase64(input:ArrayBuffer): string;
    export function fromBase64(input:string): ArrayBuffer;
    export function md5(input:ArrayBuffer): string;
    export function crc32(input:ArrayBuffer): number;
}

/** This module provides Sciter's built-in data persistence - data storage and retrieval. */
declare module '@storage' {
  declare interface storage {
    /** Storage root object containing the data.
    * any modification of this object will result in saving the data to storage file. */      
    root: any;
    /**
    * Create Index to provide effective access and ordering of potentially large data sets.
    * @param type defines type of keys in the index. It can be "string", "integer", "long", "float" or "date".
    * @param unique `true` if the index supports only unique keys, or `false` if records with the same key values are allowed in the index.
    */
    createIndex(type: 'string'|'integer'|'long'|'float'|'date', unique?: boolean): index|null;
    /** Closes underlying Storage object. Commits all data before closing.
     * After closing the storage all persistent objects that are still in use are set to non-persistent state. */
    close(): void;
    /** Commits (writes) all persistent objects reachable from its root into storage. */
    commit(): void;

    /** 
     * Registers class (a.k.a. constructor function in terms of ES5) of persistable objects.
      When an object is stored into DB, name of its class is also stored. 
      When the object is fetched from the DB, it gets the class assigned automatically if that class was registered before.
      @version 5.0.2.4+
    */
    registerClass(cls);
  }
  /** Index object in persistent storage. provide effective access and ordering of potentially large data sets. */
  interface index {
    /** Insert or replace object in index associated with the key value. */
    set(key: any, value: any, replace?: boolean): boolean;
    /** Returns object associated with the key or null. key has to be of the same type as the type of the index object.
    *  If the index was created as non unique then the return value is an array - list of items under the key. */
    get(key: any): any|any[]|null;
    /** Returns selection in the Index based on criteria `min-key, max-key` sorted by ascent or descent order, start-inclusive and end-inclusive.  
     * @info Either `minKey or maxKey` can be `null` that means search from very first or very last key in the index. */
    select(minKey: any, maxKey: any, ascending?: true, startInclusive?: true, endInclusive?: true): any[];
    /** Remove object by it key from the index. If the index is unique, obj is optional. */
    delete(key: any, object?: any): boolean;
    /** Removes all items from the index object. */
    clear(): void;

    /** Number of objects associated represented by the index. */
    readonly length: number;
    /** `true` if the index was declared as unique. */
    readonly unique: boolean;
    /** key type as it was declared at creation time. */
    readonly type: string;
  }

  /** Opens the storage and returns an instance of Storage object. If allowWrite is false then storage is opened in read-only mode. */
  export function open(path: string, allowWrite?: true): storage|null;
}

declare module "@sys" {
    export function spawn(args: string[], options?: spawnOptions ): Process;
    export function hrtime(): bigint;
    export function gettimeofday(): number;
    export function uname(): unameObject;
    /** Returns `true` if fd is an open file descriptor referring to a terminal. */
    export function isatty(): boolean;
    /** Retrieves all environment variables */
    export function environ(): object;
    export function getenv(name: string): string;
    export function setenv(name: string, value: string): void;
    export function unsetenv(name: string): void;
    export function cwd(): string;
    export function homedir(): string;
    export function tmpdir(): string;
    /** Return path of this executable file. */
    export function exepath(): string;
    export function random(buffer: ArrayBuffer);

    declare var UDP: {
        new(): UDPSocket;
    };
    declare var TCP: {
        new(): TCPSocket;
    };
    declare var TTY: {
        new(): TTY;
    };
    declare var Pipe: {
        new(): Pipe;
    };

    namespace fs {
        /**
         * Monitor files or folders for changes
         * @param path 
         * @param cb callback function
         */
        function watch(path:string, cb: (path:string, events: 0x01 | 0x02) => void): WatchFS;
        /**
         * Open file instance
         * @param path 
         * @param flags method to open the file with read/write
         * @param mode file content encoding
         */
        function open(path:string, flags: keyof typeof OpenFlagOptions, mode ?: number): Promise<File>;
        /**
         * Synchronously open file instance
         * @param path 
         * @param flags method to open the file with read/write
         * @param mode file content encoding
         * @deprecated >5.0.0.5 use `fs.sync.open()`
         */
        function $open(path:string, flags: keyof typeof OpenFlagOptions, mode ?: number): File;
        function openSync(path:string, flags: keyof typeof OpenFlagOptions, mode ?: number): File;
        /** Return information about the file at path. */
        function stat(path:string): Promise<StatStruct>;
        /** Return information about the file at path. (sync) 
         * @deprecated >5.0.0.5 use `fs.sync.stat()` or `fs.statSync()`
        */
        function $stat(path:string): StatStruct;
        /** Return information about the file at path. (sync)
         * @version 5.0.0.6+
         */
        function statSync(path:string): StatStruct;
        /** `lstat()` is identical to `stat()`, except that if path is a symbolic link, then the link itself is stat-ed, not the file that it refers to. */
        function lstat(): Promise<StatStruct>;
        /** ( sync version of `lstat()` ) `$lstat()` is identical to `$stat()`, except that if path is a symbolic link, then the link itself is stat-ed, not the file that it refers to. 
         * @deprecated >5.0.0.5 use `fs.sync.lstat()` or `fs.lstatSync()`
        */
        function $lstat(): StatStruct;
        /** ( sync version of `lstat()` ) `$lstat()` is identical to `$stat()`, except that if path is a symbolic link, then the link itself is stat-ed, not the file that it refers to. 
         * @version 5.0.0.6+
        */
        function lstatSync(): StatStruct;
        /** Expands all symbolic links and resolves references `/./`, `/../` and extra `/` characters in the pathname string to produce a canonicalized absolute pathname. */
        function realpath(pathname: string): string;
        /** Splits path to `0`: directory without trailling `/` and `1`: file name and extension */
        function splitpath(path: string): [directory: string, file: string];
        /** Remove file */
        function unlink(path:string): Promise;
        function unlinkSync(path:string);
        function rename(oldpath:string, newpath: string) : Promise;
        /** Creates unique temporary directory. The last six characters of template must be "XXXXXX". */
        function mkdtemp(template:string) : Promise<string>;
        /** Creates unique temporary file. The last six characters of template must be "XXXXXX" */
        function mkstemp(template:string) : Promise<string>;
        /** Delete directory (async) */
        function rmdir(path:string) : Promise;
        /** Delete directory (sync)
         * @deprecated >5.0.0.5 use `fs.sync.rmdir()` or `fs.rmdirSync()`
         */
        function $rmdir(path:string);
        /** Delete directory (sync) 
         * @version 5.0.0.6+
        */
        function rmdirSync(path:string);
        /** Create directory (async) */
        function mkdir(path:string, mode ?: 0o777): Promise;
        /** Create directory (sync)
         * @deprecated >5.0.0.5 use `fs.sync.mkdir()` or `fs.mkdirSync()`
         */
        function $mkdir(path:string, mode ?: 0o777);
        /** Create directory (sync)
         * @version 5.0.0.6+
         */
        function mkdirSync(path:string, mode ?: 0o777);
        /** Change mode command used to manage file system access permissions on Unix and Unix-like systems.
         * @version 5.0.0.6+
         * @reference [chmod](https://man7.org/linux/man-pages/man2/chmod.2.html)
         */
        function chmod(path: string, mode ?: number): Promise;
        /** Change mode command used to manage file system access permissions on Unix and Unix-like systems. (sync)
         * @version 5.0.0.6+
         * @reference [chmod](https://man7.org/linux/man-pages/man2/chmod.2.html)
         */
        function chmodSync(path: string, mode ?: number);
        /**
         * Asynchronous file copy.
         * @param flag a combination of `fs.UV_FS_COPYFILE_***`
         */
        function copyfile(source: string, destination: string, flag?: number): Promise;
        function copyfileSync(source: string, destination: string, flag?: number);
        /** Read directory contents asynchronously. The promise resolves to file list. */
        function readdir(path: string): Promise<FileList[]>;
        /** Read directory contents synchronously. return file list. 
         * @deprecated >5.0.0.5 use `fs.sync.readdir()` or `fs.readdirSync()`
        */
        function $readdir(path: string): FileList[];
        /** Read directory contents synchronously. return file list. 
         * @version 5.0.0.6+
        */
        function readdirSync(path: string): Promise<FileList[]>;
        /** Return file content, check `readfileSync` for sync method. */
        function readFile(path: string): Promise<ArrayBuffer>;
        function readfile(path: string): Promise<ArrayBuffer>;
        /** Synchronously return file content.
         * @deprecated >5.0.0.5 use `fs.sync.readfile()` or `fs.readfileSync()`
         */
        function $readfile(path: string): ArrayBuffer;
        /** Synchronously return file content.
         * @version 5.0.0.6+
         */
        function readFileSync(path: string): ArrayBuffer;
        function readfileSync(path: string): ArrayBuffer;
        
        const UV_DIRENT_UNKNOWN: 0;
        const UV_DIRENT_FILE: 1;
        const UV_DIRENT_DIR : 2;
        const UV_DIRENT_LINK: 3;
        const UV_DIRENT_FIFO: 4;
        const UV_DIRENT_SOCKET: 5;
        /** Character stream device, like terminal. */
        const UV_DIRENT_CHAR: 6;
        const UV_DIRENT_BLOCK: 7;
        
        /** `fs.copyfile()` flag : return an error if the destination already exists */
        const UV_FS_COPYFILE_EXCL: 1;
        /** `fs.copyfile()` flag : attempt to create a reflink, if copy-on-write is not supported, a fallback copy mechanism is used. */
        const UV_FS_COPYFILE_FICLONE: 2;
        /** `fs.copyfile()` flag : attempt to create a reflink, if copy-on-write is not supported, an error is returned. */
        const UV_FS_COPYFILE_FICLONE_FORCE: 4;

        /** Synchronous methods of existing Asynchronous file system methods.
         * @version 5.0.0.6+
         */
        namespace sync {
            /**
             * Open file instance
             * @param path 
             * @param flags method to open the file with read/write
             * @param mode file content encoding
             */
            function open(path:string, flags: keyof typeof OpenFlagOptions, mode ?: number): File;
            /** Return information about the file at path. */
            function stat(path:string): StatStruct;
            /** `lstat()` is identical to `stat()`, except that if path is a symbolic link, then the link itself is stat-ed, not the file that it refers to. */
            function lstat(): StatStruct;
            /** Delete directory. */
            function rmdir(path:string);
            /** Create directory.*/
            function mkdir(path:string, mode ?: 0o777);
            /** Change mode command used to manage file system access permissions on Unix and Unix-like systems.
             * @reference [chmod](https://man7.org/linux/man-pages/man2/chmod.2.html)
            */
            function chmod(path: string, mode ?: number);
            /** Read directory contents. The promise resolves to file list. */
            function readdir(path: string): FileList[];
            /** Return file content. */
            function readfile(path: string): ArrayBuffer;
            function unlink(path:string);
            function rename(oldpath:string, newpath: string);
            function copyfile(source: string, destination: string, flag?: number);
        }
    }

    interface Dir {
        close();
        path: string;
        next();
        [async iterator];
    }

    declare interface File {
        read (length?:number, fileposition ?: number): Promise<Uint8Array>;
        /** @deprecated >5.0.0.5 use `readSync()` */
        $read(length?:number, fileposition ?: number): Uint8Array;
        /** @version 5.0.0.5+ */
        readSync(length?:number, fileposition ?: number): Uint8Array;
        write (data:string|string[]|ArrayBuffer, filePosition ?: number) : Promise<number>;
        /** @deprecated >5.0.0.5 use `writeSync` */
        $write(data:string|string[]|ArrayBuffer, filePosition ?: number) : number;
        /** @version 5.0.0.6+ */
        writeSync(data:string|string[]|ArrayBuffer, filePosition ?: number) : number;
        close (): Promise<undefined>;
        /** @deprecated >5.0.0.5 use `closeSync` */
        $close(): undefined;
        /** @version 5.0.0.6+ */
        closeSync(): undefined;
        fileno(): number;
        stat(): Promise<StatStruct>;
        path: string;
    }

    interface FileList {
        /** local file name + extension (relative to directory) */
        name: string;
        /** ORed flags (see `fs.UV_DIRENT_****`)  */
        type: number;
    }

    declare interface WatchFS {
        readonly path: string;
        close(): void;
    }

    declare interface StatStruct {
        isFile ?: boolean;
        isDirectory ?: boolean;
        isSymbolicLink ?: boolean;
        st_dev: number;
        st_ino: number;
        st_mode: number;
        st_nlink: number;
        st_uid: number;
        st_gid: number;
        st_rdev: number;
        st_size: number;
        st_blksize: number;
        st_blocks: number;
        st_atime: number;
        st_mtime: number;
        st_ctime: number;
        st_birthtime: number;
    }

    interface unameObject {
        /** OS code name: Windows_NT */
        sysname: string;
        /** OS version: 10.0.19044  */
        release: string;
        /** OS type: Windows 10 Enterprise */
        version: string;
        /** Processor type: i686 */
        machine: string;
    }

    interface spawnOptions {
        stdin ?: string;
        stdout?: string;
        stderr?: string;
    }
}

declare enum OpenFlagOptions { 'a', 'ax', 'a+', 'ax+', 'as', 'as+', 'r', 'r+', 'rs+', 'w', 'wx', 'w+', 'wx+' }

declare interface Process {
    kill();
    wait(): Promise<ProcessStats>;
    pid: number;
    stdin: Pipe;
    stdout: Pipe;
    stderr: Pipe;
}

declare interface ProcessStats {
    exit_status: number;
    term_signal: number;
    exitCode: number;
    terminationSignal: number;
}

declare interface Socket {
    close(): void;
    fileno(): number;
    getsockname(): NetworkParam;
    getpeername(): NetworkParam;
    connect(param: NetworkParam): Promise<void>;
    bind(param: NetworkParam): void;
}

declare interface UDPSocket extends Socket {
    recv(): Promise<{data: ArrayBuffer, flags: number, addr: NetworkParam}>;
    send(data: string|ArrayBuffer): Promise<void>;
}

declare interface TCPSocket extends Socket {
    shutdown(): void;
    listen(): void;
    accept(): Promise<TCPSocket>;
    read(): Promise<ArrayBuffer>;
    write(data: string|ArrayBuffer): void;
}

declare interface Pipe extends Socket {
    listen(): void;
    accept(): Promise<Pipe>;
    bind(name: string): void;
    getpeername(): string;
    getsockname(): string;
    read(): Promise<ArrayBuffer>;
    write(data: string|ArrayBuffer): void;
}

declare interface TTY {
    close();
    read();
    write();
    fileno();
    setMode();
    getWinSize();
}

declare interface NetworkParam {
    family?: number;
    ip: string;
    port: number;
}

interface Node {
    /** Instance of Window that hosts this node; */
    readonly parentWindow: Window;
    /** Returns the previous sibling. */
    readonly previousSibling: Node | null;
    readonly nodeIndex: number;
    remove(): void;

    /** NATIVE */

    /** Returns the children. */
    readonly childNodes: NodeListOf<Node>;
    /** Returns the first child. */
    readonly firstChild: Node | null;
    /** Returns the last child. */
    readonly lastChild: Node | null;
    /** Returns the next sibling. */
    readonly nextSibling: Node | null;
    /** Returns a string appropriate for the type of node. */
    readonly nodeName: string;
    /** Returns the type of node:  
     * `1` : Element  
     * `2` : Comment  
     * `3` : Text
    */
    readonly nodeType: number;
    nodeValue: string | null;
    /** Returns the node document. Returns null for documents. */
    readonly ownerDocument: Document | null;
    /** Returns the parent element. */
    readonly parentElement: Element | null;
    /** Returns the parent. */
    readonly parentNode: Node | null;
    /** Textual content of an element and all its descendants */
    textContent: string | null;
    appendChild<T extends Node>(node: T): T;
    /** Returns a copy of node. If deep is true, the copy also includes the node's descendants. */
    cloneNode(deep?: boolean): Node;
    /** Returns a bitmask indicating the position of other relative to node. */
    compareDocumentPosition(other: Node): number;
    /** Returns true if other is an inclusive descendant of node, and false otherwise. */
    contains(other: Node | null): boolean;
    /** Returns node's root. (\<html/>) */
    getRootNode(options?: GetRootNodeOptions): Node;
    /** Does this node have children. */
    hasChildNodes(): boolean;
    insertBefore<T extends Node>(node: T, child: Node | null): T;
    /** Does this node and otherNode have the same properties. */
    isEqualNode(otherNode: Node | null): boolean;
    isSameNode(otherNode: Node | null): boolean;
    removeChild<T extends Node>(child: T): T;
    replaceChild<T extends Node>(node: Node, child: T): T;
}
declare var Node: {
    new(): Node;
};

interface Text extends Node
{
    data: string;
    readonly length: number;
    readonly wholeText: string;
}
declare var Text: {
    new(): Text;
}

interface Comment extends Node
{
    data: string;
    readonly length: number;
}
declare var Comment: {
    new(): Comment;
}

/** NodeList objects are collections of nodes, usually returned by properties such as Node.childNodes and methods such as document.querySelectorAll(). */
interface NodeList {
    /** Returns the number of nodes in the collection. */
    readonly length: number;
    /** Returns the node with index index from the collection. The nodes are sorted in tree order. */
    item(index: number): Node | null;
    /**
     * Performs the specified action for each node in an list.
     * @param callbackfn  A function that accepts up to three arguments. forEach calls the callbackfn function one time for each element in the list.
     * @param thisArg  An object to which the this keyword can refer in the callbackfn function. If thisArg is omitted, undefined is used as the this value.
     */
    forEach(callbackfn: (value: Node, key: number, parent: NodeList) => void, thisArg?: any): void;
    [index: number]: Node;
}
declare var NodeList: {
    new(): NodeList;
};

interface NodeListOf<TNode extends Node> extends NodeList {
    item(index: number): TNode;
    /**
     * Performs the specified action for each node in an list.
     * @param callbackfn  A function that accepts up to three arguments. forEach calls the callbackfn function one time for each element in the list.
     * @param thisArg  An object to which the this keyword can refer in the callbackfn function. If thisArg is omitted, undefined is used as the this value.
     */
    forEach(callbackfn: (value: TNode, key: number, parent: NodeListOf<TNode>) => void, thisArg?: any): void;
    [index: number]: TNode;
}


interface Range
{
    /** `true` if selection is collapsed to one position (anchor === focus) */
    readonly isCollapsed: boolean;
    /** Nearest container element that encloses as anchor as focus positions */
    readonly commonAncestorContainer: Element;
    readonly endContainer: Node;
    readonly endOffset: number;
    readonly startContainer: Node;
    readonly startOffset: number;
    readonly start: [node: Node, offset: number];
    readonly end: [node: Node, offset: number];

    setStart(node: Node, offset: number): void;
    setEnd(node: Node, offset: number): void;
    setStartBefore(node: Node): void;
    setEndBefore(node: Node): void;
    setStartAfter(node: Node): void;
    setEndAfter(node: Node): void;
    selectNode(node: Node): void;
    selectNodeContents(node: Node): void;
    selectNodeContent(node: Node): void;
    getRangeAt(index: number): Range;
    /** Set cursor to the start or end of selection. */
    collapse(toStart?: boolean): void;
    cloneRange(): Range;

    /** Apply marks to the selected range */
    applyMark(name: string|string[]): void;
    /** Apply marks to the selected range */
    highlight(name: string|string[]): void;
    /** Remove marks applied to the selected range */
    clearMark(name: string|string[]);
    /** Remove marks applied to the selected range */
    clearHighlight(name: string|string[]): void;
    /** Return list of the applied mark names inside the range */
    marks(): string[];
    /** Set the range to the start-end of character having the given mark name */
    setToMark(name: string): void;
}
declare var Range: {
    new(): Range;
    new(start: number, end: number): Range;
}

interface Window {
    /** Window state: hidden, shown, fullscreen...  
     *  use `Window.WINDOW_****` */
    state: number;

    /** Window has input focus. */
    readonly isActive: boolean;
    /** The property is false when the window was closed and destroyed. */
    readonly isAlive: boolean;
    /** True if window is on active space now.
     * the property is undefined if host system does not support spaces (virtual desktops). */
    readonly isOnActiveSpace: boolean|undefined;
    
    /** Get/Set Window title */
    caption: string;
    isResizable: boolean;
    isMaximizable: boolean;
    isMinimizable: boolean;
    /** Window is alway on top */
    isTopmost: boolean;
    /** Does the window accept user input. */
    isEnabled: boolean;
    /** Width to height ratio to keep on window resizes */
    aspectRatio: number;
    
    /** If set by element, direct all UI events to that element and its children. */
    eventsRoot: Element|null;
    focus: Element;
    readonly parent: Window|null;
    readonly document: Document;
    /** Parameters provided by constructor, available inside the window as they are. */
    parameters: any;

    /** Monitor index where the current window is on */
    readonly screen: number;
    /** current graphics backend used: `direct2d`, `Skia/OpenGL`, etc. */
    readonly graphicBackend: string;
    /** blur-behind effect  
     * one of `none` `auto` `dark` `ultra-dark` `light` `ultra-light` */
    blurBehind: "none" | "auto" | "dark" | "ultra-dark" | "light" | "ultra-light";

    /** Minimal size of resizable window `[width, height]` */
    minSize: [width: number, height: number];
    /** Maximum size of resizable window `[width, height]` */
    maxSize: [width: number, height: number];

    frameType: keyof typeof frameType;

    /** The function allows to enumerate elements in tab order.
     * reference element must be selectable.  
     * to select element use `window.this.focus = element`
    */
    focusable(direction: "next"|"prior"|"first"|"last", reference: Element): Element;
    /** Set input focus to window */
    activate(bringToFront: boolean): void;
    /** Request to update the window. */
    update(): void;
    /** Report geometry and data of the screen (monitor) the window is on. */
    screenBox(type: keyof typeof screenBoxType, property: "xywh"|"rect"|"position"|"dimension", asPpx?: boolean): number[];
    screenBox(type: keyof typeof screenBoxType, property: keyof typeof boxProperties, asPpx?: boolean): number;
    /** Return name of device */
    screenBox(type: 'device'): string;
    /** Is window on primary monitor. */
    screenBox(type: 'isPrimary'): boolean;
    /** Return screenshot of the monitor the window is on. */
    screenBox(type: 'snapshot'): Image;
    /** Report geometry of the window.  
     *  @param property value type to return.
     *  @param metric value in relation to.
     *  @param relativeTo offset x/y are relative to.
     *  @param asPpx return coordinates in screen pixels otherwise DIPs.
     */
    box(property: "xywh"|"rect"|"position"|"dimension", metric: keyof typeof windowBoxMetric, relativeTo?: keyof typeof windowBoxRelativeTo, asPpx?: boolean): number[];
    box(property: keyof typeof boxProperties, metric: keyof typeof windowBoxMetric, relativeTo?: keyof typeof windowBoxRelativeTo, asPpx?: boolean): number;
    /** move/size window.  
     * x, y, width, height are in PPX (physical screen pixels).  
     * If `client` is provided then parameters are window client area coordinates. */
    move(x: number, y: number, width?: number, height?: number, client?: boolean): void;
    /** move/size window to particular monitor.  
     * x, y, width, height are in DIPs - device independent pixels (a.k.a. CSS pixels). */
    moveTo(monitor: number, x: number, y: number, width?: number, height?: number, client?: boolean): void;

    /** Subscribe to window related events, init callback everytime the event occurs. */
    addEventHandler(event: windowEvent, handler: function): Window;
    /** Subscribe to window related events, init callback everytime the event occurs. */
    on(event: windowEvent, cb: eventFunction): Window;
    /** Unsubscribe from event by eventname or handler used by `on()` */
    off(eventOrHandler: windowEvent|function): Window;
    /** Send event to the window synchronously. Returns `true` if the window consumes the event. */
    dispatchEvent(event: Event): boolean;
    /** Post event to the window asynchronously. The function returns immediately - does not wait for the event consumption. */
    postEvent(event: Event): void;

    /** Load HTML document to Window */
    load(url: string): void;
    /** Open file selection dialog */
    selectFile<T extends selectFileParams>(params: T): T extends {'mode': 'open-multiple'} ? string[] : string;
    selectFile(mode: "save"|"open", filter: string): string;
    selectFile(mode: "open-multiple", filter: string): string[];
    /** Open folder selection dialog */
    selectFolder(params: object): string;

    /** Performs system event(s) in application message queue, mode is one of:  
     * `wait` - waits for the next event and executes it;  
     * `noWait` - if next event is available executes it otherwise returns immediately;  
     * `untilMouseUp` - executes events until mouseup event arrives, used for various drag cases;  
     * `untilQuit` - performs run loop - executes all events until application quit message arrives;  
     * `I/O` - performs events associated with I/O; */
    doEvent(mode?: "wait"|"noWait"|"untileMouseUp"|"untilQuit"|"I/O");
    /** Interaction with native behaviors attached to the window. */
    xcall(name: string, ...args): any;
    /** Performs drag-and-drop using system D&D mechanism. */
    perfromDrag(data: dragParams, mode: "copy"|"move", dragIcon: Image|Element,
                dragIconXoff?: number, dragIconYoff?: number): null|"copy"|"move";

    /** Show tray icon with the image and tooltip text.  
     * Tray icon will generate "trayiconclick" event for Window on user clicks */
    trayIcon({image: Image, text: string}): boolean;
    /** Remove tray icon */
    trayIcon(command: "remove"): boolean;
    /** Report location of the icon on desktop, coordinates are in screen pixels. */
    trayIcon(command: "place"): [x: number, y: number, w: number, h: number];
    /** Request user attention by flashing or bouncing window icon in task/dock bar. */
    requestAttention(command: "info" | "alert" | "stop"): void;

    /** gets/sets media variable that can be used in CSS as `@media name {...}` */
    mediaVar(name: string, value?: string): string|number|void;
    /** gets/sets multiple media variables. that can be used in CSS as `@media name {...}`*/
    mediaVars(values?: object): object|void;
    /** Show a new window as dialog, returns
     * close value of `Window.this.close(valueToReturn)` call inside the window.  
     * [Documentation](https://gitlab.com/sciter-engine/sciter-js-sdk/-/blob/main/docs/md/Window.md#windowmodaljsx-any)*/
    modal(params: windowParam): any;
    modal(params: JSX): any;

    /** Close this window and return the given value to parent window. */
    close(value?: string): boolean;
}

declare var Window: {
    new(param?: windowParam): Window;
    readonly this: Window;
    /** List of Sciter windows in the current process */
    readonly all: Window[];
    /** share is an object shared between all documents and windows in the application.
     * CAUTION: use it responsibly.If temporary window or document populates
     * shared object then it should clean it in `beforeunload` document event. */
    share: any;
    /** Number of monitors in the system */
    readonly screens: number;
    /** Report geometry and information of the given screen (monitor). */
    screenBox(screen: number, type: keyof typeof screenBoxType|'devicePixelRatio', property?: "xywh"|"rect"|"position"|"dimension"): number[];
    screenBox(screen: number, type: keyof typeof screenBoxType|'devicePixelRatio', property?: keyof typeof boxProperties): number;
    /** Return name of device */
    screenBox(screen: number, type: 'device'): string;
    /** Is this screen the primary screen. */
    screenBox(screen: number, type: 'isPrimary'): boolean;
    /** Return screenshot of the monitor. */
    screenBox(screen: number, type: 'snapshot'): Image;
    /** Return DOM element under screenX/screenY position.  
     * @info this method may return DOM element belonging to any Sciter window in current process. */
    elementAt(x: number, y: number): Element;
    /** Return value of internal timer. */
    ticks(): number;
    /** Post global event to all windows in current process. */
    post(event: Event): void;
    /** Synchronously sends global event to all windows in current process.  
     * Sending stops on first window that will consume the event by returning true from event handler of this event. */
    send(event: Event): void;
    /** Get command line arguments in a scapp app */
    scapp: {argv: string[]};


    readonly POPUP_WINDOW  : 2,
    readonly TOOL_WINDOW   : 3,
    readonly CHILD_WINDOW  : 4, // undefined
    readonly FRAME_WINDOW  : 5,
    readonly DIALOG_WINDOW : 6,

    readonly WINDOW_SHOWN      : 1;
    readonly WINDOW_MINIMIZED  : 2;
    readonly WINDOW_MAXIMIZED  : 3;
    readonly WINDOW_HIDDEN     : 4;
    readonly WINDOW_FULL_SCREEN: 5;
}

interface windowParam {
    /** Windows type use `Window.****_WINDOW` */
    type?: number;
    /** When owner closed or minimized this window will be closed/minimized too. */
    parent?: Window;
    /** window caption (or title) */
    caption?: string;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    /** x,y,w,h are coordinates of desired window client box on the screen; */
    client?: boolean;
    /** [1 to 9] alignment of the window on monitor,
     * if [-1 to -9] and parent is provided then it aligns the window against parent window.
     * (`1` bottom left corner, `2` bottom middle, `3` bottom right corner,
     * `4` middle left, `5` center, `6` middle right, `7` top left corner,
     * `8` top middle, `9` top right corner) */
    alignment?: number;
    /** index of monitor to spawn on. */
    screen?: number;
    /** Window state: hidden, shown, fullscreen...  
     *  use `Window.WINDOW_****` */
    state?: number;
    /** window html source file */
    url?: string;
    /** extra parameters to pass to the new window. */
    parameter?: any;
}

type windowEvent = "statechange" | "resolutionchange" | "mediachange" | "activate" | "replacementstart" | "replacementend" | "move" | "size" | "trayiconclick" | "spacechange";

enum frameType { "standard", "solid", "solid-with-shadow", "extended", "transparent" }

enum windowBoxMetric { 'border', 'client', 'cursor', 'caret' }
enum windowBoxRelativeTo { 'desktop', 'monitor', 'self' }
enum screenBoxType { 'frame', 'workarea', 'device', 'isPrimary', 'snapshot' }

interface selectFileParams {
    mode?: "save"|"open"|"open-multiple";
    /** File type filter, as "title|ext1;ext2".  
     *  i.e. `"HTML File (*.htm,*.html)|*.html;*.htm|All Files (*.*)|*.*"` */
    filter?: string;
    /** Default file extension. */
    extension?: string;
    /** Dialog title, "Save As" */
    caption?: string;
    /** Initial directory to open the dialog at. */
    path?: string;
}

interface dragParams {
    text?: string;
    html?: string;
    /** Single or multiple file names; */
    file?: string|string[];
    /** Any data that can be `JSON.stringify`'ed; */
    json: any;
}

