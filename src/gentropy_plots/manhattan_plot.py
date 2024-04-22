"""Plotly manhattan plot for genome-wide association studies (GWAS)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, cast

import pandas as pd
import plotly.graph_objects as go
from gentropy.common.spark_helpers import calculate_neglog_pvalue
from gentropy.dataset.summary_statistics import SummaryStatistics
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f


@dataclass
class ManhattanPlot:
    """A collection of methods to create manhattan plot from Gentropy SummaryStatistics dataset.


    Chromosome lenght metadata:
    Data accessed from: https://www.ncbi.nlm.nih.gov/grc/human/data
    Human genome build: GRCh38.p14
    """

    CHROMOSOMES: List[Tuple[str, int]] = field(
        default_factory=lambda: [
            ("1", 248956422),
            ("2", 242193529),
            ("3", 198295559),
            ("4", 190214555),
            ("5", 181538259),
            ("6", 170805979),
            ("7", 159345973),
            ("8", 145138636),
            ("9", 138394717),
            ("10", 133797422),
            ("11", 135086622),
            ("12", 133275309),
            ("13", 114364328),
            ("14", 107043718),
            ("15", 101991189),
            ("16", 90338345),
            ("17", 83257441),
            ("18", 80373285),
            ("19", 58617616),
            ("20", 64444167),
            ("21", 46709983),
            ("22", 50818468),
            ("X", 156040895),
            ("Y", 57227415),
        ]
    )

    # Default color scheme:
    EVEN_COLOR: str = "#7D9EC0"
    ODD_COLOR: str = "#71C671"
    LEAD_COLOR: str = "#b22222"

    # Manhattan plot parameters:
    PVAL_CUTOFF: float = 0.001
    APPLY_CLUMPING: bool = False

    # Marker sizes:
    DOT_SIZE: float = 3.0
    LEAD_DOT_SIZE: float = 7.0

    def _get_chromosome_annotation(self: ManhattanPlot) -> DataFrame:
        """Get chromosome annotation data.

        Returns:
            DataFrame: chromosome annotation data
        """
        # Column name in the returned dataframe:
        columns = ["chromosome", "cumulative_length", "chromosome_lenght", "color"]

        # Calculating cumulative length:
        chromosome_data: list = []

        # Compute chromosome data:
        for i, chromosome in enumerate(self.CHROMOSOMES):
            chromosome_data.append(
                (
                    chromosome[0],
                    sum([c[1] for c in self.CHROMOSOMES[:i]]),
                    chromosome[1],
                    self.EVEN_COLOR if i % 2 else self.ODD_COLOR,
                )
            )

        # Get running spark session:
        spark = SparkSession.getActiveSession()

        # Return dataframe if spark is available:
        if not spark:
            raise ValueError("No active spark session found.")

        return spark.createDataFrame(chromosome_data, columns)

    def _process_data(
        self: ManhattanPlot,
        summary_stats: SummaryStatistics,
    ) -> DataFrame:
        """Process summary statistics data.

        Args:
            summary_stats (SummaryStatistics): genome-wide single point association summary statistics

        Returns:
            DataFrame: processed summary statistics
        """
        # Extract chromosome data:
        chromosome_data = self._get_chromosome_annotation()

        # Combine with summary statistics:
        processed_sumstats = (
            summary_stats.pvalue_filter(self.PVAL_CUTOFF)
            .df.join(chromosome_data, on="chromosome", how="inner")
            .select(
                "chromosome",
                (f.col("position") + f.col("cumulative_length")).alias("position"),
                calculate_neglog_pvalue(
                    f.col("pValueMantissa"), f.col("pValueExponent")
                ).alias("negLogPValue"),
                "color",
                "studyId",
                "variantId",
                "pValueMantissa",
                "pValueExponent",
                "beta",
                f.lit(self.DOT_SIZE).alias("dotSize"),
            )
            .orderBy(f.col("position").asc())
        )

        # Do we need to combine with clumping:
        if self.APPLY_CLUMPING:
            processed_sumstats = (
                processed_sumstats
                # Joining with clumping:
                .join(
                    self._process_clumping(summary_stats),
                    on=["studyId", "variantId"],
                    how="left",
                )
                # Update colors for hits:
                .withColumn(
                    "color",
                    f.when(f.col("isIndex"), f.lit(self.LEAD_COLOR)).otherwise(
                        f.col("color")
                    ),
                )
                # Update dot size for hits:
                .withColumn(
                    "dotSize",
                    f.when(f.col("isIndex"), f.lit(self.LEAD_DOT_SIZE)).otherwise(
                        f.col("dotSize")
                    ),
                )
            )

        return processed_sumstats

    def _process_clumping(
        self: ManhattanPlot, summary_stats: SummaryStatistics
    ) -> DataFrame:
        """Process clumping data.

        Args:
            summary_stats (SummaryStatistics): genome-wide single point association summary statistics

        Returns:
            DataFrame: clumping data
        """
        return summary_stats.window_based_clumping().df.select(
            "studyId", "variantId", f.lit(True).alias("isIndex")
        )

    def create_manhattan_plot(
        self: ManhattanPlot, summary_statistics: SummaryStatistics
    ) -> go.Figure:
        """Generate manhattan plot from summary statistics.

        Args:
            self (ManhattanPlot): _description_
            summary_statistics (SummaryStatistics): genome-wide single point association summary statistics

        Returns:
            go.Figure:
        """
        # Do data processing prior plotting:
        processed_summary_statistics = cast(
            pd.DataFrame, self._process_data(summary_statistics).toPandas()
        )

        study_chromosomes = processed_summary_statistics.chromosome.unique()
        study_id = processed_summary_statistics.studyId[0]

        # Extract data to show:
        custom_data = processed_summary_statistics[
            ["variantId", "pValueMantissa", "pValueExponent", "beta"]
        ].to_numpy()

        # Create hover template:
        hover_template = (
            ",".join(
                [
                    "<BR><b>VariantId: </b> %{customdata[0]}",
                    "<BR><b>P-value: </b> %{customdata[1]:.3f}E%{customdata[2]}",
                    "<BR><b>Beta: </b> %{customdata[3]:.3f}",
                ]
            )
            + "<extra></extra>"
        )

        # X-axis limits +/- 2% of the full range:
        x_axis_min = (
            processed_summary_statistics.position.min()
            - (
                processed_summary_statistics.position.max()
                - processed_summary_statistics.position.min()
            )
            * 0.02
        )
        x_axis_max = (
            processed_summary_statistics.position.max()
            + (
                processed_summary_statistics.position.max()
                - processed_summary_statistics.position.min()
            )
            * 0.02
        )
        # Get chromosome labels and their positions:
        x_axis_ticks = [
            x["x"]
            for x in self._get_chromosome_annotation()
            .withColumn(
                "x", f.col("cumulative_length") + f.col("chromosome_lenght") / 2
            )
            .collect()
            if x["chromosome"] in study_chromosomes
        ]
        x_axis_tick_labels = [
            x["chromosome"]
            for x in self._get_chromosome_annotation().collect()
            if x["chromosome"] in study_chromosomes
        ]

        # Create figure:
        fig = (
            go.Figure()
            # Adding single point associations:
            .add_trace(
                go.Scatter(
                    customdata=custom_data,
                    hovertemplate=hover_template,
                    x=processed_summary_statistics.position,
                    y=processed_summary_statistics.negLogPValue,
                    mode="markers",
                    marker=go.scatter.Marker(
                        size=processed_summary_statistics.dotSize,
                        color=processed_summary_statistics.color,
                        opacity=0.8,
                        line=dict(width=0),
                    ),
                )
            )
            .update_layout(
                # Adding title:
                title=study_id,
                title_x=0.5,
                # Clear background
                plot_bgcolor="white",
                # Add custom x-axis labels:
                xaxis=dict(
                    tickmode="array",
                    # Get the x-axis position where the chromosome labels should be placed:
                    tickvals=x_axis_ticks,
                    # Get chromosome labels:
                    ticktext=x_axis_tick_labels,
                    range=[x_axis_min, x_axis_max],
                ),
            )
            # Updating y-axis:
            .update_yaxes(
                ticks="outside",
                showline=True,
                linecolor="black",
                showgrid=False,
                title="-log10(p-value)",
                mirror=True,
            )
            # Update x-axis:
            .update_xaxes(
                ticks="outside",
                tickangle=90,
                showgrid=False,
                showline=True,
                mirror=True,
                linecolor="black",
            )
            # Adding line showing significance:
            .add_shape(
                type="line",
                layer="below",
                y0=7.3,
                y1=7.3,
                x0=x_axis_min,
                x1=x_axis_max,
                line=dict(color="firebrick", width=1),
            )
        )

        return fig
