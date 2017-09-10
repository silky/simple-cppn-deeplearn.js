function buildModel (graph, batchSize, latentDim, w, h, scale) {
	var outputCanvas = document.getElementById("output");
    var netSize      = 16;
    var colours      = 3;
	var pixels       = w * h;

	var z = graph.placeholder("latent_z", [ batchSize
										  // HACK: Due to lack of broadcasting. We perform the broadcasting
										  // in the input space, instead of here.
										  , pixels
										  , latentDim
									      ]);
	var x = graph.placeholder("x", [batchSize * pixels, 1]);
	var y = graph.placeholder("y", [batchSize * pixels, 1]);
	var r = graph.placeholder("r", [batchSize * pixels, 1]);

	function ones (shape) {
		var r = deeplearn.Array2D.zeros(shape);
		var cr = graph.constant(r);
		var c1 = graph.constant(1);
		var added = graph.add(cr, c1);
		return added;
	}

	function fc (name, input, size, activation, includeBias) {
		// TODO: Why can't we use layers.dense?

		if( !includeBias ){
			includeBias = false;
		}

		var vals    = deeplearn.Array2D.randNormal([input.shape[1], size]);
		var weights = graph.variable("weights", vals);
		var result  = graph.matmul(input, weights);

		if( includeBias ){
			bias   = deeplearn.Array2D.randNorml([shape[0], size]);
			bias   = graph.variable("bias", bias);
			result = graph.add(weights, bias);
		}

		return result;
	}

	var z_scaled = graph.multiply( z 
							     , graph.multiply(ones([1, pixels, latentDim]), graph.constant(scale))
							 	 );
	var z_unrolled = graph.reshape(z_scaled, [batchSize * pixels, latentDim]);
	var x_unrolled = x;
	var y_unrolled = y;
	var r_unrolled = r;

	U = fc("g_0_z", z_unrolled, netSize, graph.relu);
	U = graph.add(U, fc("g_0_x", x_unrolled, netSize, graph.relu, false));
	U = graph.add(U, fc("g_0_y", y_unrolled, netSize, graph.relu, false));
	U = graph.add(U, fc("g_0_r", r_unrolled, netSize, graph.relu, false));

	H = graph.tanh(U);

	for(k = 0; k <= 3; k++){
		H = graph.tanh(fc("g_tanh_" + k, H, netSize, graph.relu));
	}

	net = graph.sigmoid(fc("g_tanh_" + k, H, colours, graph.relu));
	net = graph.reshape(net, [w, h, colours]);

	return [net, z, x, y, r];
}

// What the heck, JavaScript.
function range (k, f) {
	r = new Array(k);
	for(i = 0; i < r.length; i++){
		r[i] = f(i);
	}
	return r;
}

function mathOnes (math, shape) {
	var r = deeplearn.Array2D.zeros(shape);
	return math.add(r, deeplearn.Scalar.ONE);
}

function vectorInputs (math, w, h, scale) {
	if( !scale ){
		scale = 1;
	}

	function f(dim, x ) {
		var g = function (x) {
			return (x - (dim - 1) / 2) / (dim - 1) / 0.5
		};
		return g;
	}

	var pixels = w * h;
	var xRange = deeplearn.Array1D.new(range(w, f(w)));
	var yRange = deeplearn.Array1D.new(range(h, f(h)));

	var xMat = math.matMul(mathOnes(math, [h, 1]), xRange.reshape([1, w]));
	var yMat = math.matMul(yRange.reshape([h, 1]), mathOnes(math, [1, w]));

	var rMat = math.sqrt( math.add( math.multiply(xMat, xMat)
					              , math.multiply(yMat, yMat) ) );

	xMat = xMat.reshape([pixels, 1]);
	yMat = yMat.reshape([pixels, 1]);
	rMat = rMat.reshape([pixels, 1]);

	return [xMat, yMat, rMat];
}

function forward (net, session, math, z_, feeds, ctx, w, h, batchSize, latentDim) {
	var zvec   = deeplearn.Array2D.randUniform([batchSize, 1, latentDim], -1, 1);
	zvec       = math.multiply(zvec, mathOnes(math, [1, w*h, latentDim]));

	var zFeeds = feeds.concat([{"tensor": z_, "data": zvec }]);

	vals = session.eval(net, zFeeds);
	vals = vals.getValues();

	var img = ctx.getImageData(0, 0, w, h);

	for(x = 0; x < w; x++){
		for(y = 0; y < h; y++){
			var ix = (y + (x * w))*4;
			var iv = (y + (x * w))*3;

			img.data[ix + 0] = Math.floor(255 * vals[iv + 0]);
			img.data[ix + 1] = Math.floor(255 * vals[iv + 1]);
			img.data[ix + 2] = Math.floor(255 * vals[iv + 2]);
			img.data[ix + 3] = 255;
		}
	}

	ctx.putImageData(img, 0, 0);
	
	// requestAnimationFrame( function () {
	// 	math.scope(function (){
	// 		forward( net, session, math, z_, feeds, ctx, w, h, batchSize, latentDim );
	// 	});
	// });
}

function go () {
	var canvas = document.getElementById("output");
	var ctx    = canvas.getContext("2d");
	var graph     = new deeplearn.Graph;
	var batchSize = 1;
	var latentDim = 8;
    var w         = canvas.width; 
    var h         = canvas.height;
	var scale     = 1;

	var math = new deeplearn.NDArrayMathGPU();

	math.scope(function () {
		[net, z_, x_, y_, r_] = buildModel(graph, batchSize, latentDim, w, h, scale);

		[xvec, yvec, rvec] = vectorInputs(math, w, h, scale);

		var feeds = [ {"tensor": x_, "data": xvec}
					, {"tensor": y_, "data": yvec}
					, {"tensor": r_, "data": rvec}
					];

		var session = new deeplearn.Session(graph, math);

		forward( net, session, math, z_, feeds, ctx, w, h, batchSize, latentDim );
	});
}
