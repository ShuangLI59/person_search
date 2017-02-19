$(document).ready(function() {

var IMAGE_ROOT = '../data/psdb/dataset/Image/SSM/';
var CANVAS_HEIGHT = 300;

// Helper function to stretch a canvas
function stretchCanvas(canvas, ratio) {
  canvas.style.height = CANVAS_HEIGHT + 'px';
  canvas.style.width = Math.round(CANVAS_HEIGHT * ratio) + 'px';
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}

// Display results
function displayResults(data) {
  var curIndex = 0;

  var showImage = function(fname, bbox, canvasId, correctness) {
    var fpath = IMAGE_ROOT + fname;
    var img = new Image();
    img.onload = function() {  // show image and bbox
      // canvas for whole scene
      var canvas = document.getElementById(canvasId);
      var ctx = canvas.getContext('2d');
      // resize the image and canvas
      var imgRatio = img.width / img.height;
      stretchCanvas(canvas, imgRatio);
      var r = Math.min(canvas.width / img.width, canvas.height / img.height);
      var offset = Math.floor((canvas.width - Math.floor(img.width * r)) / 2);
      ctx.drawImage(img, offset, 0,
                    Math.floor(img.width*r),
                    Math.floor(img.height*r));
      // draw bounding box
      var pad = 10;
      var x = bbox[0] - pad, y = bbox[1] - pad,
          width = bbox[2] - bbox[0] + 2*pad,
          height = bbox[3] - bbox[1] + 2*pad;
      ctx.lineWidth = 7;
      ctx.strokeStyle = ['#FF5722', '#4CAF50', '#2196F3'][correctness];
      ctx.strokeRect(offset+Math.floor(x*r), Math.floor(y*r),
                     Math.floor(width*r), Math.floor(height*r));
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#FFFFFF';
      ctx.strokeRect(offset+Math.floor(x*r), Math.floor(y*r),
                     Math.floor(width*r), Math.floor(height*r));
    };
    img.src = fpath;
  };

  // Append or remove canvas HTML elements for gallery
  var initCanvasList = function(divId, num) {
    var div = $('#' + divId);
    while (div.children().length < num) {
      div.append('<canvas id="' + divId + '-canvas-' +
                 div.children().length + '"></canvas>')
    }
    $('#' + divId + ' canvas:gt(' + (num-1) + ')').remove();
  };

  // Render results on the current index
  var render = function() {
    var item = data.results[curIndex];
    // update header status bar
    $('#cur-id-text').val(curIndex + 1);
    $('#counter').text(' / ' + data.results.length);
    // probe and gt
    initCanvasList('probe', item.probe_gt.length + 1);
    showImage(item.probe_img, item.probe_roi, 'probe-canvas-0', 2);
    for (var i = 0; i < item.probe_gt.length; ++i) {
      showImage(item.probe_gt[i].img, item.probe_gt[i].roi,
                'probe-canvas-' + (i+1), 1);
    }
    // gallery
    initCanvasList('gallery', item.gallery.length);
    for (var i = 0; i < item.gallery.length; ++i) {
      showImage(item.gallery[i].img, item.gallery[i].roi,
                'gallery-canvas-' + i, item.gallery[i].correct);
    }
  };
  render();

  // User interactions
  var prev = function() {
    if (curIndex > 0) {
      --curIndex;
      render();
    }
  };
  var next = function() {
    if (curIndex + 1 < data.results.length) {
      ++curIndex;
      render();
    }
  };
  var rand = function() {
    curIndex = Math.floor(Math.random() * (data.results.length - 1));
    render();
  };
  var jump = function() {
    var idx = parseInt($('#cur-id-text').val());
    curIndex = Math.max(0, Math.min(data.results.length - 1, idx - 1));
    render();
  };

  document.onkeydown = function(e) {
    var mapping = {
      65: prev,
      68: next,
      82: rand,
    };
    if (e.keyCode in mapping) mapping[e.keyCode]();
  };

  $('#prev-btn').click(prev);
  $('#next-btn').click(next);
  $('#rand-btn').click(rand);
  $('#go-btn').click(jump);
  $('#cur-id-text').keypress(function(e) {
    if (e.which === 13) jump();
  });
};

// Load json results file
function loadResults() {
  var fileURL = 'results.json?sigh=' +
                 Math.floor(Math.random() * 100000);  // prevent caching
  $.getJSON(fileURL, function(data) {
    displayResults(data);
  });
};
loadResults();

});