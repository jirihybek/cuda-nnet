/*
 * Spiking neural network
 *
 * Viewer scripts
 */

var graph = function(canvas, network){

	var layerMargin = 130;
	var nodeMargin = 40;
	var nodeRadius = 10;
	var nodeEvenOffset = 35;
	var canvasMargin = 50;

	var nodeMax = 0;

	//Calcualte size
	for(var l in network)
		nodeMax = Math.max(nodeMax, network[l].length);

	var canvasWidth = canvasMargin * 2 + network.length * layerMargin;
	var canvasHeight = canvasMargin * 2 + nodeMax * nodeMargin;
	var centerY = Math.round(canvasHeight / 2);

	canvas.setAttribute("width", canvasWidth);
	canvas.setAttribute("height", canvasHeight);

	var context = canvas.getContext("2d");

	var render = function(){

		context.beginPath();
		context.rect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
		context.fillStyle = '#222222';
		context.fill();

		//Render connections
		context.lineWidth = 2;
		context.lineCap = 'round';

		for(var l in network){
			for(var n in network[l]){

				var node = network[l][n];
				var nodeX = canvasMargin + l * layerMargin + ( n % 2 == 1 ? nodeEvenOffset : 0 );
				var nodeY = centerY - (network[l].length / 2) * nodeMargin + n * nodeMargin;

				for(var i in node.inputs){

					var input = node.inputs[i];

					var targetX = canvasMargin + input.layer * layerMargin + ( input.node % 2 == 1 ? nodeEvenOffset : 0 );
					var targetY = centerY - (network[input.layer].length / 2) * nodeMargin + input.node * nodeMargin;

					if(network[input.layer][input.node].value * input.weight > 0)
						context.strokeStyle = 'rgba(0, 255, 0, ' + (0.3 + Math.abs(input.weight / 2)) + ')';
					else if(network[input.layer][input.node].value * input.weight < 0)
						context.strokeStyle = 'rgba(255, 0, 0, ' + (0.3 + Math.abs(input.weight / 2)) + ')';
					else
						context.strokeStyle = 'rgba(255, 255, 255, ' + (0.3 + Math.abs(input.weight / 2)) + ')';

					context.beginPath();
					context.moveTo(nodeX, nodeY);
					context.lineTo(targetX, targetY);
					context.stroke();

				}

			}
		}

		//Render layers
		context.lineWidth = 4;

		for(var l in network){
			for(var n in network[l]){

				var node = network[l][n];
				var nodeX = canvasMargin + l * layerMargin + ( n % 2 == 1 ? nodeEvenOffset : 0 );
				var nodeY = centerY - (network[l].length / 2) * nodeMargin + n * nodeMargin;

				//Draw node body
				context.beginPath();
				context.arc(nodeX, nodeY, nodeRadius, 0, 2 * Math.PI, false);
				context.fillStyle = '#ffffff';
				context.fill();

				//Draw sum indicator
				if(node.value > 0){
					context.beginPath();
					context.arc(nodeX, nodeY, nodeRadius + 2, 0, 2 * Math.PI, false);
					context.fillStyle = '#CDDC39';
					context.strokeStyle = '#9E9D24';
					context.fill();
					context.stroke();
				} else if(node.value < 0){
					context.beginPath();
					context.arc(nodeX, nodeY, nodeRadius + 2, 0, 2 * Math.PI, false);
					context.fillStyle = '#F44336';
					context.strokeStyle = '#D32F2F';
					context.fill();
					context.stroke();
				} else if(node.sum > 0){
					context.beginPath();
					context.arc(nodeX, nodeY, nodeRadius * (node.sum / node.threshold), 0, 2 * Math.PI, false);
					context.fillStyle = '#2196F3';
					context.fill();
				}

			}
		}

	};

	render();

	return render;

};

var applyState = function(network, state){

	for(var l in state){
		for(var n in state[l]){

			network[l][n].value = state[l][n].v;
			network[l][n].sum = state[l][n].s;

			for(var i in state[l][n].i)
				network[l][n].inputs[i].weight = state[l][n].i[i].w;

		}
	}

};

var setup = function(network, states){

	var toolbar      = document.getElementById("toolbar");
	var stepsControl = document.getElementById("steps");
	var prevControl  = document.getElementById("prev");
	var nextControl  = document.getElementById("next");
	var pauseControl = document.getElementById("pause");
	var playControl  = document.getElementById("play");

	stepsControl.setAttribute("max", states.length);
	stepsControl.value = 0;

	var timer = null;

	//Create graph
	var update = graph(document.getElementById("canvas"), network);

	//State update
	var setState = function(index){
		applyState(network, states[index]);
		update();
	};

	var prev = function(){
		stepsControl.value = Math.max(0, parseInt(stepsControl.value) - 1);
		setState(parseInt(stepsControl.value));
	};

	var next = function(){
		stepsControl.value = Math.min(states.length, parseInt(stepsControl.value) + 1);
		setState(parseInt(stepsControl.value));
	};

	var reset = function(){
		stepsControl.value = 0;
		setState(parseInt(0));
	};

	var play = function(){
		
		if(timer) return;

		if(parseInt(stepsControl.value) >= states.length)
			reset();

		timer = setInterval(function(){
			if(parseInt(stepsControl.value) >= states.length)
				return stop();

			next();
		}, 50);

		toolbar.classList.add("playing");

	};

	var stop = function(){

		if(timer) clearInterval(timer);
		timer = null;

		toolbar.classList.remove("playing");

	};

	//Bind events
	prevControl.addEventListener("click", function(){
		prev();
	});

	nextControl.addEventListener("click", function(){
		next();
	});

	stepsControl.addEventListener("change", function(){
		setState(parseInt(stepsControl.value));
	});

	playControl.addEventListener("click", function(){
		play();
	});

	pauseControl.addEventListener("click", function(){
		stop();
	});

};

function loadDump(src, done){

	var el = document.createElement("script");

	el.onerror = function(){
		return done(true);
	};

	el.onload = function(){
		return done(null);
	};

	el.type = "text/javascript";
	el.src = src;

	document.getElementsByTagName("head")[0].appendChild(el);

}

function requestDump(done){

	var src = prompt("Dump filename", "dump.js");

	if(!src) return;

	loadDump(src, function(err){

		if(err){
			alert("Cannot load dump file.");
			requestDump();
		}

		if(done) done();

	});

}

window.addEventListener("load", function(){

	//requestDump();
	requestDump(function(){

		setup(network, states);

	});

});