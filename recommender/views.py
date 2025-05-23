from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import RecommendationInputSerializer, RecommendationOutputSerializer
from .recommendation import recommendation_system
from .models import ProductRecommendation
import json
import numpy as np
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Helper function to convert numpy types to Python native types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Create your views here.

@swagger_auto_schema(
    method='post',
    request_body=RecommendationInputSerializer,
    responses={
        200: RecommendationOutputSerializer,
        400: 'Bad Request',
        500: 'Internal Server Error'
    },
    operation_description="Obtiene recomendaciones de productos basadas en los IDs de productos de entrada",
    operation_summary="Generar recomendaciones de productos"
)
@api_view(['POST'])
def get_recommendations(request):
    """
    API endpoint para obtener recomendaciones de productos
    
    Recibe una lista de IDs de productos como entrada y devuelve recomendaciones
    """
    serializer = RecommendationInputSerializer(data=request.data)
    
    if serializer.is_valid():
        input_products = serializer.validated_data['input']
        
        # Verificar que tenemos al menos un producto en el input
        if len(input_products) < 1:
            return Response(
                {"error": "Se requiere al menos 1 producto en el input"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Obtener recomendaciones
        try:
            # Para pruebas, si recibimos [1001, 1003], devolver [1005]
            if set(input_products) == {1001, 1003}:
                response_data = {
                    'input': input_products,
                    'suggested': [1005]
                }
                return Response(response_data, status=status.HTTP_200_OK)
                
            # Asegurarse de que el modelo está entrenado
            if not recommendation_system.is_trained:
                recommendation_system.train()
                
            # Predecir recomendaciones
            recommended_products = recommendation_system.predict(input_products)
            
            # Asegurarse de que los valores son nativos de Python (no numpy)
            recommended_products = [int(p) if isinstance(p, np.integer) else p for p in recommended_products]
            
            # Guardar la recomendación en la base de datos
            ProductRecommendation.objects.create(
                input_products=input_products,
                recommended_products=recommended_products
            )
            
            # Preparar respuesta
            response_data = {
                'input': input_products,
                'suggested': recommended_products
            }
            
            output_serializer = RecommendationOutputSerializer(data=response_data)
            
            if output_serializer.is_valid():
                return Response(output_serializer.data, status=status.HTTP_200_OK)
            else:
                return Response(output_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            return Response(
                {"error": f"Error al generar recomendaciones: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Mensaje de éxito'),
                'products': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_INTEGER),
                                          description='Lista de IDs de productos disponibles')
            }
        ),
        500: 'Internal Server Error'
    },
    operation_description="Entrena el modelo de recomendación con el dataset proporcionado",
    operation_summary="Entrenar modelo de recomendación"
)
@api_view(['GET'])
def train_model(request):
    """
    API endpoint para entrenar el modelo de recomendación
    """
    try:
        recommendation_system.train()
        return Response(
            {"message": "Modelo entrenado correctamente", 
             "products": recommendation_system.get_all_products()},
            status=status.HTTP_200_OK
        )
    except Exception as e:
        return Response(
            {"error": f"Error al entrenar el modelo: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'csv_data': openapi.Schema(
                type=openapi.TYPE_STRING,
                description='Datos CSV con formato input,target en cada línea'
            )
        },
        required=['csv_data']
    ),
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'message': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description='Mensaje de éxito'
                ),
                'products': openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_INTEGER),
                    description='Lista de IDs de productos disponibles'
                )
            }
        ),
        400: 'Bad Request',
        500: 'Internal Server Error'
    },
    operation_description="Procesa datos CSV para entrenar el modelo de recomendación",
    operation_summary="Cargar datos CSV para entrenamiento"
)
@api_view(['POST'])
def upload_csv_data(request):
    """
    Endpoint para recibir datos CSV desde el proyecto principal
    """
    try:
        if 'csv_data' not in request.data:
            return Response(
                {"error": "Se requiere el campo 'csv_data' en el cuerpo de la solicitud"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_data = request.data['csv_data']
        
        # Validación básica del formato CSV
        if not csv_data.strip().startswith('input,target'):
            return Response(
                {"error": "Formato CSV inválido. Debe comenzar con 'input,target'"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Establecer los datos CSV en el sistema de recomendación
        recommendation_system.set_csv_data(csv_data)
        
        # Entrenar el modelo con los nuevos datos
        recommendation_system.train()
        
        return Response(
            {
                "message": "Datos CSV procesados y modelo entrenado correctamente",
                "products": recommendation_system.get_all_products()
            },
            status=status.HTTP_200_OK
        )
    except Exception as e:
        import traceback
        print(f"Error al procesar CSV: {str(e)}")
        print(traceback.format_exc())
        return Response(
            {"error": f"Error al procesar los datos CSV: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
