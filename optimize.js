/*global angular*/

angular.module('main', []);

angular.module('main').controller('optimize', ['$scope', '$http', function ($scope, $http) {
    $scope.charts = [];

    $http({ url: '/optimize/charts' }).then(
        function (res) {
            layout = {xaxis: {}, yaxis: {}};
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

            trace = {line: {}};
            trace.y = res.data;
            trace.type = 'scatter';
            trace.mode = 'lines';
            trace.line.color = 'darkblue';
            trace.line.width = 1.5;
            $scope.charts = [{layout: layout, data: [trace], id: 1}];
        },

        function (trace) {        
            $scope.error = trace;
        }
    );
}]);