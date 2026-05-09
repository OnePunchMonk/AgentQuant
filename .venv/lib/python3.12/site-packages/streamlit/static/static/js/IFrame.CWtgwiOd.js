import{o as e}from"./chunk.DuxOD-Sk.js";import{r as t}from"./emotion-is-prop-valid.esm.DV_OErv-.js";import{c as n,t as r}from"./emotion-styled.browser.esm.BPUx6WuE.js";import{$ as i,Q as a}from"./protobuf.DyZGXwGk.js";import{n as o,t as s}from"./IFrameUtil.BaqCY7QW.js";var c=e(t()),l=r(`iframe`,{target:`ei36xw10`})(({theme:e,disableScrolling:t,width:n,height:r})=>({width:n??`100%`,height:r??`100%`,colorScheme:`normal`,border:`none`,padding:e.spacing.none,margin:e.spacing.none,overflow:t?`hidden`:void 0})),u=`streamlit:iframe:setSize`,d=`25rem`,f=`<script>
(function() {
  var lastW = 0, lastH = 0;
  function sendSize() {
    // Guard against malformed HTML (e.g., <frameset>) or script running before body init
    if (!document.body) return;
    // Use getBoundingClientRect for accurate fractional pixel measurement,
    // then ceil to avoid scrollbars from sub-pixel rounding
    var rect = document.body.getBoundingClientRect();
    var w = Math.ceil(Math.max(
      rect.width,
      document.body.scrollWidth,
      document.body.offsetWidth,
      document.documentElement.scrollWidth,
      document.documentElement.offsetWidth
    ));
    var h = Math.ceil(Math.max(
      rect.height,
      document.body.scrollHeight,
      document.body.offsetHeight,
      document.documentElement.scrollHeight,
      document.documentElement.offsetHeight
    ));
    if (w !== lastW || h !== lastH) {
      lastW = w; lastH = h;
      // Note: postMessage with '*' broadcasts to any origin, but this is safe because:
      // 1. This script only runs inside srcdoc (same-origin, sandboxed)
      // 2. The payload is just dimension integers
      // 3. The frontend receiver validates event.source === iframe.contentWindow
      window.parent.postMessage({type: '${u}', width: w, height: h}, '*');
    }
  }
  // Send initial size after DOM is ready
  if (document.readyState === 'complete') {
    sendSize();
  } else {
    window.addEventListener('load', sendSize);
  }
  // Re-measure on DOM changes
  if (typeof MutationObserver !== 'undefined') {
    new MutationObserver(sendSize).observe(document.body, {
      childList: true, subtree: true, attributes: true, characterData: true
    });
  }
  // Re-measure on resize and image/font loading
  window.addEventListener('resize', sendSize);
  document.addEventListener('load', sendSize, true);
})();
<\/script>`;function p(e){return e+f}function m(e){return a(e)||e===``?void 0:e}function h({element:e,widthConfig:t,heightConfig:r}){let a=m(e.src),f=i(a)?void 0:m(e.srcdoc),h=(0,c.useRef)(null),[g,_]=(0,c.useState)({width:null,height:null}),v=t?.useContent??!1,y=r?.useContent??!1,b=i(f)&&(v||y),x=b?p(f):f;(0,c.useEffect)(()=>{if(!b)return;let e=e=>{if(e.source&&e.source===h.current?.contentWindow){let t=e.data;if(t?.type===u&&typeof t?.width==`number`&&typeof t?.height==`number`&&Number.isFinite(t.width)&&Number.isFinite(t.height)&&t.width>=0&&t.height>=0){let e=t.width,n=t.height;_(t=>t.width===e&&t.height===n?t:{width:e,height:n})}}};return window.addEventListener(`message`,e),()=>{window.removeEventListener(`message`,e)}},[b]);let S=b&&v&&g.width!==null?`${g.width}px`:void 0,C;return y&&(b&&g.height!==null?C=`${g.height}px`:i(a)&&(C=d)),n(l,{ref:h,className:`stIFrame`,"data-testid":`stIFrame`,allow:s,disableScrolling:!e.scrolling,src:a,srcDoc:x,scrolling:e.scrolling?`auto`:`no`,sandbox:o,title:`st.iframe`,tabIndex:e.tabIndex??void 0,width:S,height:C})}var g=(0,c.memo)(h);export{g as default};