
var data, labels, N;
var dpcolor;
var ss = 40.0; // 50.0; // scale for drawing
var state = 'stop'; // state for stop, start, pause buttons

// use toggle to show or hide prediction space
var t_pred_color = false;
var t_pause = false;

// create neural net
var layer_defs, net, trainer;
var t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});\n\
layer_defs.push({type:'fc', num_neurons:4, activation: 'relu'});\n\
layer_defs.push({type:'fc', num_neurons:4, activation: 'relu'});\n\
layer_defs.push({type:'softmax', num_classes:2});\n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.001, momentum:0.1, batch_size:10, l2_decay:0.001});\n\
";

function reload() {
  eval($("#layerdef").val());

  // MJS - enter buttons for layers
  var t = '';
  for(var i=1;i<net.layers.length-1;i++) { // ignore input and regression layers (first and last)
    var butid = "button" + i;
    t += "<input class=\"btn btn-primary\" id=\""+butid+"\" value=\"" + net.layers[i].layer_type + "(" + net.layers[i].out_depth + ")" +"\" type=\"submit\" onclick=\"updateLix("+i+")\" style=\"width:80px; height: 30px; margin:5px;\";>";
  }
  $("#layer_ixes").html(t);
  $("#button"+lix).css('color', 'orange');
  
  $("#cyclestatus").html('drawing neurons ' + d0 + ' and ' + d1 + ' of layer with index ' + lix + ' (' + net.layers[lix].layer_type + ')');
  switch(net.layers[lix].layer_type) {
    case 'fc':
        $("#explanation").html('Weights and Bias rotate and skew the feature space');
      break;
      case 'relu':
          $("#explanation").html('Is this neuron firing? Data point must be non-zero');
        break;
    default:
        $("#explanation").html('nothing');
      break;
  }

  updateTitles(lix)
  
}

function playPress() {
  if(state=='stop'){
    state='play';
    var button = d3.select("#button_play").classed('btn-success', true); 
    button.select("i").attr('class', "fa fa-pause");  
  }
  else if(state=='play' || state=='resume'){
    state = 'pause';
    d3.select("#button_play i").attr('class', "fa fa-play"); 
  }
  else if(state=='pause'){
    state = 'resume';
    d3.select("#button_play i").attr('class', "fa fa-pause");        
  }
  console.log("button play pressed, play was "+state);
}

function stopPress(){
  state = 'stop';
  var button = d3.select("#button_play").classed('btn-success', false);
  button.select("i").attr('class', "fa fa-play");
  console.log("button stop invoked.");    
}

function togglepause() {
  t_pause = t_pause ? false : true;
  if(t_pause) {
    $("#buttonpause").val('resume');
  } else {
    $("#buttonpause").val('pause');
  }
  console.log('pause is ' + t_pause);
}

function updateTitles(lix) {
  // MJS - update graph titles
  var hidden_layer_count = (net.layers.length - 3) / 2; 
  var weight_v_act = lix % 2 === 1 ? ' weighted inputs' : ' activation function';
  var layer_text = (lix === net.layers.length -2) ? 'Output Layer' : ' Hidden Layer ' + (Math.floor((lix-1)/2) + 1) + ' - ' + weight_v_act;
  $("#layerlabel").html(layer_text);
  if(lix === net.layers.length -2) {
    $("#leftaxislabel").html('         P(dog)');
    $("#topaxislabel").html('(0,0) P(cat)');
  } else {
    $("#leftaxislabel").html('neuron ' + d0 );
    $("#topaxislabel").html('(0,0) neuron ' + d1 );
  }
 
}

function updateLix(newlix) {
  $("#button"+lix).css('color', 'white'); // erase highlight
  lix = newlix;
  d0 = 0;
  d1 = 1; // reset these
  $("#button"+lix).css('color', 'orange');  // highlight selected button in orange

 // MJS - update cycle and explanation text
  $("#cyclestatus").html('drawing neurons ' + d0 + ' and ' + d1 + ' of layer with index ' + lix + ' (' + net.layers[lix].layer_type + ')');
  switch(net.layers[lix].layer_type) {
    case 'fc':
      if(lix == net.layers.length - 2) {
        // this is the output layer
        $("#explanation").html('How well does the prediction fit the data points?');
      } else {
        $("#explanation").html('Weights and Bias rotate and skew the feature space');
      }
        
      break;
      case 'relu':
          $("#explanation").html('Is this neuron firing? Data point must be non-zero');
        break;
    default:
        $("#explanation").html('nothing');
      break;
  }

  updateTitles(lix);
}
 

function myinit() { }

function random_data(){
  data = [];
  labels = [];
  for(var k=0;k<40;k++) {
    data.push([convnetjs.randf(-3,3), convnetjs.randf(-3,3)]); labels.push(convnetjs.randf(0,1) > 0.5 ? 1 : 0);
  }
  N = labels.length;
}

function original_data(){
  
  data = [];
  labels = [];
  data.push([-0.4326  ,  1.1909 ]); labels.push(1);
  data.push([3.0, 4.0]); labels.push(1);
  data.push([0.1253 , -0.0376   ]); labels.push(1);
  data.push([0.2877 ,   0.3273  ]); labels.push(1);
  data.push([-1.1465 ,   0.1746 ]); labels.push(1);
  data.push([1.8133 ,   1.0139  ]); labels.push(0);
  data.push([2.7258 ,   1.0668  ]); labels.push(0);
  data.push([1.4117 ,   0.5593  ]); labels.push(0);
  data.push([4.1832 ,   0.3044  ]); labels.push(0);
  data.push([1.8636 ,   0.1677  ]); labels.push(0);
  data.push([0.5 ,   3.2  ]); labels.push(1);
  data.push([0.8 ,   3.2  ]); labels.push(1);
  data.push([1.0 ,   -2.2  ]); labels.push(1);
  N = labels.length;

  original_color();
}

function original_color(){
  dpcolor = [];
  labels.forEach(l => {
    dpcolor.push(0);
  })
  // dpcolor[2] = 1;
}

function update_color(idx){
  dpcolor[idx] = dpcolor[idx] == 1 ? 0 : 1;
}

function circle_data() {
  data = [];
  labels = [];
  for(var i=0;i<50;i++) {
    var r = convnetjs.randf(0.0, 2.0);
    var t = convnetjs.randf(0.0, 2*Math.PI);
    data.push([r*Math.sin(t), r*Math.cos(t)]);
    labels.push(1);
  }
  for(var i=0;i<50;i++) {
    var r = convnetjs.randf(3.0, 5.0);
    //var t = convnetjs.randf(0.0, 2*Math.PI);
    var t = 2*Math.PI*i/50.0
    data.push([r*Math.sin(t), r*Math.cos(t)]);
    labels.push(0);
  }
  N = data.length;

  original_color();
}

function spiral_data() {
  data = [];
  labels = [];
  var n = 100;
  for(var i=0;i<n;i++) {
    var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
    var t = 1.25*i/n*2*Math.PI + convnetjs.randf(-0.1, 0.1);
    data.push([r*Math.sin(t), r*Math.cos(t)]);
    labels.push(1);
  }
  for(var i=0;i<n;i++) {
    var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
    var t = 1.25*i/n*2*Math.PI + Math.PI + convnetjs.randf(-0.1, 0.1);
    data.push([r*Math.sin(t), r*Math.cos(t)]);
    labels.push(0);
  }
  N = data.length;

  original_color();
}
 
function update(){
  // forward prop the data

  // MJS - If stopped, reset loss_meter, or if paused,  exit
  if(state === 'stop' ) {  
    
    $("#loss_meter").html('Loss function: ');
    return; 
  } else if (state === 'pause') { return; };



  var start = new Date().getTime();

  var x = new convnetjs.Vol(1,1,2);
  //x.w = data[ix];
  var avloss = 0.0;
  for(var iters=0;iters<20;iters++) {
    for(var ix=0;ix<N;ix++) {
        x.w = data[ix];
        var stats = trainer.train(x, labels[ix]);
        avloss += stats.loss;
     
    }
  }
  avloss /= N*iters;

  // MJS - Display accumulated loss  each refresh
  $("#loss_meter").html('Loss function: ' + round(avloss,4));

  var end = new Date().getTime();
  var time = end - start;
      
  //console.log('loss = ' + avloss + ', 100 cycles through data in ' + time + 'ms');
}

function cycle() {
  var selected_layer = net.layers[lix];
  d0 += 1;
  d1 += 1;
  if(d1 >= selected_layer.out_depth) d1 = 0; // and wrap
  if(d0 >= selected_layer.out_depth) d0 = 0; // and wrap
  
  updateTitles(lix);
}

var lix = 1; // layer id to track first 2 neurons of
var d0 = 0; // first dimension to show visualized
var d1 = 1; // second dimension to show visualized
function draw(){
    
    ctx.clearRect(0,0,WIDTH,HEIGHT);
    
    var netx = new convnetjs.Vol(1,1,2);
    // draw decisions in the grid
    var density= 5.0;
    var gridstep = 2;
    var gridx = [];
    var gridy = [];
    var gridl = []; 
    for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
      for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
        //var dec= svm.marginOne([(x-WIDTH/2)/ss, (y-HEIGHT/2)/ss]);
        netx.w[0] = (x-WIDTH/2)/ss;
        netx.w[1] = (y-HEIGHT/2)/ss;
        var a = net.forward(netx, false);

        // MJS - use toggle to show or hide prediction space
        if(t_pred_color) {
          if(a.w[0] > a.w[1]) ctx.fillStyle = 'rgb(250, 150, 150)';
          else ctx.fillStyle = 'rgb(150, 250, 150)';

          //ctx.fillStyle = 'rgb(150,' + Math.floor(a.w[0]*105)+150 + ',150)';
          //ctx.fillStyle = 'rgb(' + Math.floor(a.w[0]*255) + ',' + Math.floor(a.w[1]*255) + ', 0)';
          ctx.fillRect(x-density/2-1, y-density/2-1, density+2, density+2);
        }
        if(cx%gridstep === 0 && cy%gridstep===0) {
          // record the transformation information
          var xt = net.layers[lix].out_act.w[d0]; // in screen coords
          var yt = net.layers[lix].out_act.w[d1]; // in screen coords
          gridx.push(xt);
          gridy.push(yt);
          gridl.push(a.w[0] > a.w[1]); // remember final label as well
        }
      }
    }

    // draw axes
    ctx.beginPath();
    ctx.strokeStyle = 'rgb(50,50,50)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, HEIGHT/2);
    ctx.lineTo(WIDTH, HEIGHT/2);
    ctx.moveTo(WIDTH/2, 0);
    ctx.lineTo(WIDTH/2, HEIGHT);
    ctx.stroke();

    // draw representation transformation axes for two neurons at some layer
    var mmx = cnnutil.maxmin(gridx);
    var mmy = cnnutil.maxmin(gridy);
    visctx.clearRect(0,0,visWIDTH,visHEIGHT);
    visctx.strokeStyle = 'rgb(0, 0, 0)';
    var n = Math.floor(Math.sqrt(gridx.length)); // size of grid. Should be fine?
    var ng = gridx.length;
    var c = 0; // counter
    visctx.beginPath() 
    for(var x=0;x<n;x++) {
      for(var y=0;y<n;y++) {

        // down
        var ix1 = x*n+y;
        var ix2 = x*n+y+1;
        if(ix1 >= 0 && ix2 >= 0 && ix1 < ng && ix2 < ng && y<n-1) { // check oob
          var xraw = gridx[ix1];
          xraw1 = visWIDTH*(gridx[ix1] - mmx.minv)/mmx.dv;
          yraw1 = visHEIGHT*(gridy[ix1] - mmy.minv)/mmy.dv;
          xraw2 = visWIDTH*(gridx[ix2] - mmx.minv)/mmx.dv;
          yraw2 = visHEIGHT*(gridy[ix2] - mmy.minv)/mmy.dv;
          visctx.moveTo(xraw1, yraw1);
          visctx.lineTo(xraw2, yraw2);
        }

        // and draw its color on the vizctx canvas
        // MJS - use toggle to show or hide prediction space
        if(t_pred_color) {
          if(gridl[ix1]) visctx.fillStyle = 'rgb(250, 150, 150)';
          else visctx.fillStyle = 'rgb(150, 250, 150)';
          var sz = density * gridstep;
          visctx.fillRect(xraw1-sz/2-1, yraw1-sz/2-1, sz+2, sz+2);
        }

        // right
        var ix1 = (x+1)*n+y;
        var ix2 = x*n+y;
        if(ix1 >= 0 && ix2 >= 0 && ix1 < ng && ix2 < ng && x <n-1) { // check oob
          var xraw = gridx[ix1];
          xraw1 = visWIDTH*(gridx[ix1] - mmx.minv)/mmx.dv;
          yraw1 = visHEIGHT*(gridy[ix1] - mmy.minv)/mmy.dv;
          xraw2 = visWIDTH*(gridx[ix2] - mmx.minv)/mmx.dv;
          yraw2 = visHEIGHT*(gridy[ix2] - mmy.minv)/mmy.dv;
          visctx.moveTo(xraw1, yraw1);
          visctx.lineTo(xraw2, yraw2);
        }
 
      }
    }
    visctx.stroke();

    // draw datapoints.
    ctx.strokeStyle = 'rgb(0,0,0)';
    ctx.lineWidth = 1;
    for(var i=0;i<N;i++) {
      
      if(labels[i]==1) ctx.fillStyle = 'rgb(100,200,100)';
      else ctx.fillStyle = 'rgb(200,100,100)';

      if(dpcolor[i]==1)  { ctx.fillStyle = 'rgb(100,100,200)'; }

      drawCircle(data[i][0]*ss+WIDTH/2, data[i][1]*ss+HEIGHT/2, 5.0);
      

      // also draw transformed data points while we're at it
      netx.w[0] = data[i][0];
      netx.w[1] = data[i][1]
      var a = net.forward(netx, false);
      var xt = visWIDTH * (net.layers[lix].out_act.w[d0] - mmx.minv) / mmx.dv; // in screen coords
      var yt = visHEIGHT * (net.layers[lix].out_act.w[d1] - mmy.minv) / mmy.dv; // in screen coords
      if(labels[i]==1) visctx.fillStyle = 'rgb(100,200,100)';
      else visctx.fillStyle = 'rgb(200,100,100)';
      // MJS - Override datapoint color if corresponding dpcolor flag is set
      if(dpcolor[i]==1)  visctx.fillStyle = 'rgb(100,100,200)';
      visctx.beginPath();
      visctx.arc(xt, yt, 5.0, 0, Math.PI*2, true); 
      visctx.closePath();
      visctx.stroke();
      visctx.fill();
    }
}

function mouseClick(x, y, shiftPressed, ctrlPressed, altPressed, cmdPressed){
  // modify
  x = x // - 250;
  y = y // - 140;
  // x and y transformed to data space coordinates
  var xt = (x-WIDTH/2)/ss;
  var yt = (y-HEIGHT/2)/ss;

  if(altPressed) {
    // MJS - highlight closest data point by updating dpcolor flag
    var mink = -1;
    var mind = 99999;
    for(var k=0, n=data.length;k<n;k++) {
      var dx = data[k][0] - xt;
      var dy = data[k][1] - yt;
      var d = dx*dx+dy*dy;
      if(d < mind || k==0) {
        mind = d;
        mink = k;
      }
    }
    if(mink>=0) {
      update_color(mink)
    }

  } else if(ctrlPressed || cmdPressed) {
    // remove closest data point
    var mink = -1;
    var mind = 99999;
    for(var k=0, n=data.length;k<n;k++) {
      var dx = data[k][0] - xt;
      var dy = data[k][1] - yt;
      var d = dx*dx+dy*dy;
      if(d < mind || k==0) {
        mind = d;
        mink = k;
      }
    }
    if(mink>=0) {
      console.log('splicing ' + mink);
      data.splice(mink, 1);
      labels.splice(mink, 1);
      N -= 1;
    }

  } else {
    // add datapoint at location of click
    data.push([xt, yt]);
    labels.push(shiftPressed ? 1 : 0);
    N += 1;
  }

  

}

function keyDown(key){
}

function keyUp(key) {
}

function blink(element)
    {
        setInterval(function () {
            element.style.webkitTransitionDuration = "0.7s";
            element.style.opacity = 0;
            setTimeout(function () {
                element.style.webkitTransitionDuration = "0.7s";
                element.style.opacity = 1;
            }, 700);
        },1400);

    }

function round(value, decimals) {
  return Number(Math.round(value+'e'+decimals)+'e-'+decimals);
}

$(function() {
    // note, globals
    viscanvas = document.getElementById('viscanvas');
    visctx = viscanvas.getContext('2d');
    visWIDTH = viscanvas.width;
    visHEIGHT = viscanvas.height;

    $('#t_pred_space').change(function() {
      t_pred_color = $(this).prop('checked') ? true : false;
      console.log('Toggle: ' + $(this).prop('checked'));
    })

    circle_data();
    $("#layerdef").val(t);
    reload();
    NPGinit(20);

});