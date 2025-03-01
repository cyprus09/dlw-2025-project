import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';

interface CO2SequestrationDataPoint {
    year: number;
    claimed: number;
    actual: number;
}

interface CO2ComparisonChartProps {
    data: CO2SequestrationDataPoint[];
    title?: string;
}

const CO2SequestrationChart: React.FC<CO2ComparisonChartProps> = ({
    data,
    title = 'CO2 Sequestration Comparison',
}) => {
    return (
        <div className='w-full'>
            <ResponsiveContainer
                width='100%'
                height={220}
            >
                <LineChart
                    data={data}
                    margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
                >
                    <CartesianGrid strokeDasharray='3 3' />
                    <XAxis dataKey='year' />
                    <YAxis
                        label={{
                            value: 'CO2 (tons)',
                            angle: -90,
                            position: 'insideLeft',
                        }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line
                        name='Claimed Sequestration'
                        type='monotone'
                        dataKey='claimed'
                        stroke='#8884d8'
                        strokeWidth={2}
                    />
                    <Line
                        name='Actual Sequestration'
                        type='monotone'
                        dataKey='actual'
                        stroke='#82ca9d'
                        strokeWidth={2}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default CO2SequestrationChart;

// Example usage:
// const sampleData = [
//     { year: 2019, claimed: 150, actual: 120 },
//     { year: 2020, claimed: 200, actual: 160 },
//     { year: 2021, claimed: 250, actual: 200 },
//     { year: 2022, claimed: 300, actual: 230 },
// ];
//
// <CO2SequestrationChart data={sampleData} />
