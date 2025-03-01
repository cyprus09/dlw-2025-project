import React from 'react';
import InsightWidget from '@/components/widgets/InsightWidget';
import { Navbar } from '@/components/Navbar';

import mapImage from '../assets/map-img.png';
import vegImage from '../assets/veg-img.png';
import CO2SequestrationChart from '@/components/CO2SequestrationChart';

const YearlyComparison = ({
    year,
    claimed,
    actual,
}: {
    year: string;
    claimed: string;
    actual: string;
}) => {
    return (
        <div className='flex flex-col'>
            <div className='text-neutral-500 font-medium'>{year}</div>
            <div className='flex justify-between w-full'>
                <div>
                    <div className='text-lg font-medium text-yellow-600'>
                        {claimed}
                    </div>
                    <div>Claimed</div>
                </div>
                <div>
                    <div className='text-lg font-medium text-emerald-600'>
                        {actual}
                    </div>
                    <div>Actual</div>
                </div>
            </div>
        </div>
    );
};

const CarbonInsights: React.FC = () => {
    const riskValue = 80;
    const latitude = 27.234925;
    const longitude = -18.936622;
    const aiInsights = [
        'The proposed project area is located in a region with high carbon sequestration potential.',
        'The project area is located in a region with high biodiversity.',
        'The project area is located in a region with high deforestation rates.',
        'The project area is located in a region with high risk of wildfires.',
    ];
    const sampleData = [
        { year: 2019, claimed: 150, actual: 120 },
        { year: 2020, claimed: 200, actual: 160 },
        { year: 2021, claimed: 250, actual: 200 },
        { year: 2022, claimed: 300, actual: 230 },
    ];
    return (
        <div className='flex flex-col min-h-screen'>
            <Navbar />
            <div className='h-[calc(100vh-50px)] p-8 overflow-hidden bg-gradient-to-b from-neutral-200 to-neutral-50'>
                <div className='mb-6'>
                    <h1 className='scroll-m-20 text-3xl font-extrabold tracking-tighter lg:text-4xl mb-2 text-emerald-950'>
                        Carbon Integrity Insights
                    </h1>
                    <h2 className='text-xl font-medium mb-4 text-neutral-500'>
                        An AI-Driven Validation & Fraud Analysis Dashboard
                    </h2>
                </div>
                <div className='flex justify-between items-center gap-20 mb-8 w-full'>
                    <div className='max-w-9/12'>
                        Project summary lorem ipsum, dolor sit amet consectetur
                        adipisicing elit. Illo ullam enim nemo quidem corporis,
                        itaque officia sed, ratione consequuntur, ex impedit in
                        facere quaerat quibusdam voluptatibus facilis alias
                        nostrum distinctio incidunt quia eveniet tempore odit!
                        Lorem ipsum dolor, sit amet consectetur adipisicing
                        elit. Consequuntur nisi, cumque provident tempore vel
                        voluptatibus expedita reprehenderit ad vero aut, porro,
                        dolore ipsa error exercitationem quis itaque obcaecati
                        voluptatem facere temporibus doloribus tempora!
                    </div>
                    <div className='mr-6'>
                        <div className='w-full text-neutral-400 font-medium'>
                            Risk Meter
                        </div>
                        <div className='flex items-baseline gap-2'>
                            <div
                                className={`text-3xl font-semibold ${
                                    riskValue < 25
                                        ? 'text-green-600'
                                        : riskValue < 50
                                        ? 'text-yellow-600'
                                        : riskValue < 75
                                        ? 'text-orange-600'
                                        : 'text-red-600'
                                }`}
                            >
                                {riskValue}
                            </div>
                            <div className='text-sm mb-0.5'>/ 100</div>
                        </div>
                        <div
                            className={`font-medium ${
                                riskValue < 25
                                    ? 'text-green-600'
                                    : riskValue < 50
                                    ? 'text-yellow-600'
                                    : riskValue < 75
                                    ? 'text-orange-600'
                                    : 'text-red-600'
                            }`}
                        >
                            {riskValue < 25
                                ? 'Safe'
                                : riskValue < 50
                                ? 'Moderate'
                                : riskValue < 75
                                ? 'High'
                                : 'Unsafe'}
                        </div>
                    </div>
                </div>

                <div className='grid grid-cols-12 grid-rows-2 gap-4'>
                    {/* div1: spans rows 1-2, cols 1-3 */}
                    <div className='col-start-1 col-end-4 row-start-1 row-end-3'>
                        <InsightWidget title='Project Goals'>
                            <div className='flex flex-col gap-4 justify-evenly'>
                                <div className='text-sm'>
                                    Lorem ipsum dolor sit amet, consectetur
                                    adipiscing elit. Sed do eiusmod tempor
                                    incididunt ut labore et dolore magna aliqua.
                                    Ut enim ad minim veniam, quis nostrud
                                    exercitation ullamco laboris nisi ut aliquip
                                    ex ea commodo consequat.
                                </div>
                                <div className='text-sm'>
                                    Lorem ipsum dolor sit amet, consectetur
                                    adipiscing elit. Sed do eiusmod tempor
                                    incididunt ut labore et dolore magna aliqua.
                                    Ut enim ad minim veniam, quis nostrud
                                    exercitation ullamco laboris nisi ut aliquip
                                    ex ea commodo consequat.
                                </div>
                                <div className='rounded-md mt-2'>
                                    <img
                                        className='rounded-lg'
                                        src={mapImage}
                                        alt='map'
                                    />
                                </div>
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div2: spans row 1, cols 4-6 */}
                    <div className='col-start-4 col-end-7 row-start-1 row-end-2'>
                        <InsightWidget title='Vegetation Index'>
                            <div className='rounded-md mt-2'>
                                <img
                                    className='rounded-lg'
                                    src={vegImage}
                                    alt='map'
                                />
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div3: spans row 2, cols 4-6 */}
                    <div className='col-start-4 col-end-7 row-start-2 row-end-3'>
                        <InsightWidget title='Forest Area'>
                            <div className='flex flex-col gap-2'>
                                <YearlyComparison
                                    year='2018'
                                    claimed='65.28%'
                                    actual='76.29%'
                                />
                                <YearlyComparison
                                    year='2019'
                                    claimed='67.28%'
                                    actual='78.29%'
                                />
                                <YearlyComparison
                                    year='2020'
                                    claimed='69.28%'
                                    actual='80.29%'
                                />
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div4: spans row 1, cols 7-9 */}
                    <div className='col-start-7 col-end-10 row-start-1 row-end-2'>
                        <InsightWidget title='CO₂ Reductions'>
                            <CO2SequestrationChart data={sampleData} />
                        </InsightWidget>
                    </div>

                    {/* div5: spans row 1, cols 10-12 */}
                    <div className='col-start-10 col-end-13 row-start-1 row-end-2'>
                        <InsightWidget title='Carbon Content'>
                            <div className='flex flex-col gap-1'>
                                <div className='text-neutral-500 font-medium'>
                                    Calculated
                                </div>
                                <div className='text-emerald-600 text-xl font-medium'>
                                    224.5 tC/ha
                                </div>
                                <div className='mt-4 flex flex-col gap-2'>
                                    <div>
                                        Carbon content is calculated based on
                                        the amount of carbon stored in the soil
                                        and vegetation of the area.
                                    </div>
                                    <div>
                                        This value is used to estimate the total
                                        carbon sequestration potential of the
                                        area.
                                    </div>
                                </div>
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div6: spans row 2, cols 7-12 */}
                    <div className='col-start-7 col-end-13 row-start-2 row-end-3'>
                        <InsightWidget title='AI Insights'>
                            {/* Produce some real looking AI insights on a carbon credit proposal */}
                            {aiInsights.map((insight, index) => (
                                <div
                                    key={index}
                                    className='mb-2'
                                >
                                    💡 {insight}
                                </div>
                            ))}
                        </InsightWidget>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CarbonInsights;
