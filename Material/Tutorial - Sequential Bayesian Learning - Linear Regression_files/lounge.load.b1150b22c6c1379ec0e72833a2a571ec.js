!function(){"use strict";var a=window.document,b={STYLES:"https://c.disquscdn.com/next/embed/styles/lounge.5eddbf623f13f2ed755bba2f86cff840.css",RTL_STYLES:"https://c.disquscdn.com/next/embed/styles/lounge_rtl.a6759545d55b4c100fb835d64cb901b2.css","lounge/main":"https://c.disquscdn.com/next/embed/lounge.bundle.722b0c6527f60483ba4bc985cdde0821.js","recommendations/main":"https://c.disquscdn.com/next/embed/recommendations.bundle.17ad366c5f5c89b6bc1ae1ec83966565.js","remote/config":"https://disqus.com/next/config.js","common/vendor_extensions/highlight":"https://c.disquscdn.com/next/embed/highlight.6fbf348532f299e045c254c49c4dbedf.js"};window.require={baseUrl:"https://c.disquscdn.com/next/current/embed/embed",paths:["lounge/main","recommendations/main","remote/config","common/vendor_extensions/highlight"].reduce(function(a,c){return a[c]=b[c].slice(0,-3),a},{})};var c=a.createElement("script");c.onload=function(){require(["common/main"],function(a){a.init("lounge",b)})},c.src="https://c.disquscdn.com/next/embed/common.bundle.a0ed109e21af94c55c513d7580d5773c.js",a.body.appendChild(c)}();