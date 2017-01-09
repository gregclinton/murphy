/*global angular,Plotly*/

angular.module('main', []);

angular.module('main').controller('main', ['$scope', '$http', function ($scope, $http) {
    $scope.charts = [];
    
    function error(trace) {
        $scope.error = trace;
    }
    
    function polyline(x, y, color) {
        return {x: x, y: y, mode: 'lines', line: { color: color } };
    }

    function dot(x, y, size, color) {
        return {x: [x], y: [y], mode: 'markers', marker: { size: size, color: color } };
    }

    function layout(width, height) {
        var layout = { xaxis: {}, yaxis: {} };
        
        layout.width = width;
        layout.height = height;
        layout.showlegend = false;
        layout.xaxis.showticklabels = false;
        layout.xaxis.ticks = '';
        layout.xaxis.autorange = true;
        layout.xaxis.zeroline = false;
        layout.xaxis.showgrid = false;
        layout.yaxis.autorange = true;
        layout.yaxis.showticklabels = false;
        layout.yaxis.ticks = '';
        layout.yaxis.zeroline = false;
        layout.yaxis.showgrid = false;
        layout.margin = {t: 2, l: 1, r: 1, b: 2};
        return layout;
    }

    $scope.drill = function(id) {
        alert(id);
    };

    $scope.next = function () {
        $http({ url: 'optimize/next' }).then(
            function (res) {
                Plotly.deleteTraces('chart', 0);
                Plotly.addTraces('chart', { y: res.data, mode: 'lines', line: {color: 'darkred'} });
            }, error);
    };

    $http({ url: 'optimize/charts' }).then(
        function (res) {
            var traces = [],
                contour = res.data.contour,
                options = { displayModeBar: false, staticPlot: true };

            min = function (a) { return Math.min.apply(null, a); };
            max = function (a) { return Math.max.apply(null, a); };
            start = min(contour.z.map(min));
            end = max(contour.z.map(max));

            contour.type = 'contour';
            contour.contours = { coloring: 'lines', start: start, end: end, size: (end - start) / 5.0 };
            contour.showscale = false;

            traces.push(contour);
            traces.push(dot(50, 50, 13, 'green'));
            
            $scope.charts = [
                {data: traces, layout: layout(80, 50), id: 1},
                {data: traces, layout: layout(80, 50), id: 2},
                {data: traces, layout: layout(80, 50), id: 3},
                {data: traces, layout: layout(80, 50), id: 4},
                {data: traces, layout: layout(80, 50), id: 5}
            ];
                              
            Plotly.plot('chart', traces, layout(320, 180), options);
        }, error);
}]);

angular.module('main').directive('chart', [function () {
    function link(scope, element, attribute) {
        var json = scope.data,
            options = { displayModeBar: false, staticPlot: true };
        
        Plotly.plot(element[0], json.data, json.layout, options);
    }    
    
    return {
        restrict: 'E',
        scope: { data: '=' },
        template: '<span style="outline: none; border: 0;"> </span>',
        link: link
    };
}]);

angular.module('main').directive('shortcut', ['$document', function ($document) {
    return {
        restrict: 'E',
        replace: true,
        scope: true,
        link: function (scope, element, attrs) {
            function code(key) {
                return key === 'enter' ? 13 : key.charCodeAt(0);
            }

            $document.bind('keypress', function (e) {
                if (code(attrs.key) === e.which) {
                    scope.$apply(attrs.keypress);
                }
            });
        }
    };
}]);