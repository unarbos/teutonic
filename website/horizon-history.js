(function () {
  'use strict';
  var data = null, loading = null, selected = null, main2 = [], refineCodeCutover = false;
  var root = document.getElementById('horizon-content');
  function escape(s) { return String(s).replace(/[&<>"']/g, function (c) { return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
  function number(n) { return n.toLocaleString('en-GB'); }
  function date(t) { return new Date(t).toLocaleDateString('en-GB', {day:'numeric',month:'short',timeZone:'UTC'}).toUpperCase(); }
  function signed(n) { return (n >= 0 ? '+' : '') + n.toFixed(5); }
  function draw() {
    if (!data) return;
    var rows = window.HorizonHistoryData.merge(data.kings, main2).filter(function(r) { return Number.isFinite(r.king) && Number.isFinite(r.horizon); }).map(function(r) { return Object.assign({},r,{gap:r.king-r.horizon}); });
    if (!rows.length) { root.textContent = 'No matched historical results available.'; return; }
    if (selected === null) selected = rows.length - 1;
    selected=Math.min(selected,rows.length-1);
    var width = Math.max(280, root.clientWidth), height = width < 500 ? 340 : 420;
    var left=48, right=width-22, top=28, floor=height-45;
    var firstMain2 = rows.findIndex(function(r) { return r.competition === 'main2'; });
    var showCutover = refineCodeCutover || firstMain2 !== -1;
    var plotEnd = showCutover && firstMain2 === -1 ? right-18 : right;
    var times=rows.map(function(r) {return Date.parse(r.timestamp);});
    var values=rows.flatMap(function(r) {return [r.king,r.horizon];});
    var low=Math.min.apply(null,values), high=Math.max.apply(null,values);
    var step=Math.max(.005, Math.ceil((high-low)/6/.005)*.005);
    low=Math.floor((low-step*.1)/step)*step;
    high=Math.ceil((high+step*.1)/step)*step;
    if(high===low) high=low+step;
    function x(i) { return rows.length === 1 ? (left+plotEnd)/2 : left+i/(rows.length-1)*(plotEnd-left); }
    function y(v) { return floor-(v-low)/(high-low)*(floor-top); }
    var partial=rows.some(function(r) {return !r.complete;});
    var html='<svg class="horizon-history" viewBox="0 0 '+width+' '+height+'" role="group" aria-label="MAIN and MAIN2 kings and Horizon loss" aria-describedby="horizon-guidance">';
    for(var v=low;v<=high+step/10;v+=step) {
      var yy=y(v);
      html+='<line class="trend-grid" x1="'+left+'" x2="'+right+'" y1="'+yy+'" y2="'+yy+'"/><text class="trend-tick" x="'+(left-8)+'" y="'+(yy+3)+'" text-anchor="end">'+v.toFixed(3)+'</text>';
    }
    html+='<line class="trend-axis" x1="'+left+'" x2="'+right+'" y1="'+floor+'" y2="'+floor+'"/>';
    var tickCount = Math.min(rows.length, 8, Math.max(2, Math.floor((plotEnd-left)/65)+1));
    for (var tick=0; tick<tickCount; tick++) {
      var index=Math.round(tick*(rows.length-1)/(tickCount-1 || 1)), tickX=x(index);
      html+='<line class="trend-axis" x1="'+tickX+'" x2="'+tickX+'" y1="'+floor+'" y2="'+(floor+4)+'"/>';
      html+='<text class="trend-tick" x="'+tickX+'" y="'+(floor+17)+'" text-anchor="middle">'+date(times[index])+'</text>';
    }
    html+='<text class="trend-axis-label" x="'+((left+right)/2)+'" y="'+(floor+34)+'" text-anchor="middle">KING CORONATIONS · UTC</text>';
    ['horizon','king'].forEach(function(role) {
      html+='<polyline class="trend-'+role+'-line" points="'+rows.map(function(r,i) {return x(i)+','+y(r[role]);}).join(' ')+'"/>';
    });
    if (showCutover) {
      var cutoverX = firstMain2 > 0 ? (x(firstMain2-1)+x(firstMain2))/2 : firstMain2 === 0 ? left : right;
      var flagWidth=126, flagX=Math.max(left,Math.min(cutoverX-flagWidth/2,right-flagWidth));
      var cutoverLabel='RefineCode added to the evaluation dataset at the MAIN2 cutover';
      html+='<g class="dataset-change-marker trend-cutover" tabindex="0" role="img" aria-label="'+cutoverLabel+'"><title>'+cutoverLabel+'</title>';
      html+='<line class="dataset-change-line" x1="'+cutoverX+'" x2="'+cutoverX+'" y1="'+top+'" y2="'+floor+'"/>';
      html+='<line class="dataset-change-hit" x1="'+cutoverX+'" x2="'+cutoverX+'" y1="'+top+'" y2="'+floor+'"/>';
      html+='<rect class="dataset-change-flag" x="'+flagX+'" y="7" width="'+flagWidth+'" height="16"/>';
      html+='<text class="dataset-change-label" x="'+(flagX+flagWidth/2)+'" y="18" text-anchor="middle">REFINECODE ADDED</text></g>';
    }
    rows.forEach(function(r,i) {
      var xx=x(i);
      var label=(r.competition || 'main').toUpperCase()+' · '+r.model+' · '+date(times[i])+' · King '+r.king.toFixed(5)+' / Horizon '+r.horizon.toFixed(5)+' bits/byte · gap '+signed(r.gap)+' · '+number(r.documents)+' of '+number(r.expected_documents)+' documents'+(r.complete?'':' · partial backfill');
      html+='<g class="trend-point" data-index="'+i+'" tabindex="0" role="button" aria-label="'+escape(label)+'"><title>'+escape(label)+'</title><rect class="trend-hit" x="'+(xx-9)+'" y="'+top+'" width="18" height="'+(floor-top)+'"/><line class="trend-pair" x1="'+xx+'" x2="'+xx+'" y1="'+y(r.king)+'" y2="'+y(r.horizon)+'"/><circle class="trend-king-dot" cx="'+xx+'" cy="'+y(r.king)+'" r="3.5"/><rect class="trend-horizon-dot" x="'+(xx-2.5)+'" y="'+(y(r.horizon)-2.5)+'" width="5" height="5"/></g>';
    });
    html+='</svg>';
    if(partial) html+='<p class="trend-caveat">Historical backfill is partial.</p>';
    root.innerHTML=html;
    function show(i) {
      root.querySelectorAll('.trend-point').forEach(function(p) {p.classList.toggle('selected',Number(p.dataset.index)===i);});
    }
    show(selected);
    root.querySelectorAll('.trend-point').forEach(function(p) {
      var i=Number(p.dataset.index);
      p.addEventListener('pointerenter',function(){show(i);});
      p.addEventListener('focus',function(){show(i);});
      p.addEventListener('pointerleave',function(){show(selected);});
      p.addEventListener('blur',function(){show(selected);});
      p.addEventListener('click',function(){selected=i;show(i);});
      p.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();selected=i;show(i);}});
    });
  }
  function render(force) {
    if(data && force !== true) return; // Dashboard polling must not reset the selected historical point.
    if(loading) return loading;
    if(!data) root.textContent='LOADING HISTORICAL RESULTS…';
    loading=fetch('horizon-history.json',{cache:'no-store'}).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}).then(function(value){data=value;draw();}).catch(function(){if(!data) root.textContent='Historical data unavailable. Reload to retry.';}).finally(function(){loading=null;});
    return loading;
  }
  window.HorizonComparison={
    render:render,
    updateMain2:function(payload) {
      main2=window.HorizonHistoryData.main2Kings(payload);
      draw();
    },
    markRefineCodeCutover:function() {
      if (refineCodeCutover) return;
      refineCodeCutover=true;
      draw();
    }
  };
  var previousWidth=0;
  new ResizeObserver(function(entries){var w=Math.round(entries[0].contentRect.width);if(w!==previousWidth){previousWidth=w;draw();}}).observe(root);
  setInterval(function(){render(true);},60000);
  render(); // The backup chart also works when live dashboard feeds are unavailable.
})();
