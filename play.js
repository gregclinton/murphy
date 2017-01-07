/*global angular,Plotly*/

angular.module('main', []);

angular.module('main').controller('main', ['$scope', '$http', function ($scope, $http) {
    $scope.charts = [];

    $http({ url: '/optimize/charts' }).then(
        function (res) {
            var layout = {xaxis: {}, yaxis: {}},
                trace = {line: {}},
                options = { displayModeBar: false, staticPlot: true };

            layout.height = 180;
            layout.width = 320;
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

            trace.y = res.data;
            trace.type = 'scatter';
            trace.mode = 'lines';
            trace.line.color = 'darkblue';
            trace.line.width = 1.5;

            Plotly.plot('chart', [trace], layout, options);
        },

        function (trace) {
            $scope.error = trace;
        }
    );
}]);

angular.module('main').directive('shortcut', ['$document', function ($document) {
    return {
        restrict: 'E',
        replace: true,
        scope: true,
        link: function (scope, element, attrs) {
            function code(key) {
                return key === 'enter' ? 13 : key === 'backspace' ? 8 : key.charCodeAt(0);
            }
            
            $document.bind('keypress', function (e) {
                if (code(attrs.key) === e.which) {
                    scope.$apply(attrs.keypress);
                }
            });
            
            $document.bind('keydown', function (e) {
                if (code(attrs.key) === e.which) {
                    scope.$apply(attrs.keydown);
                }
            });
        }
    };
}]);